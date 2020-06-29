//===-- HeapProtector.cpp - temporal heap safety ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of KernelHeapProtector, an instrumentation pass and heap
// implementation to provide temporal heap safety for the Linux kernel.
// Details of the algorithm:
//   TODO: write.
//
//===----------------------------------------------------------------------===//

#include "llvm/InitializePasses.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BreadthFirstIterator.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/SwapByteOrder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Utils/SimplifyLibCalls.h"
#include <algorithm>
#include <iomanip>
#include <limits>
#include <list>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_map>

using namespace llvm;

static cl::opt<uint32_t> PercpuPinHeadOffset(
    "heap-protector-percpu-offset",
    cl::desc(
        "Override the location of the pin list head in the percpu segment"),
    cl::init(0x20));
static cl::opt<uint32_t> PercpuPinHeadAS(
    "heap-protector-percpu-as",
    cl::desc(
        "Override the address space number used for percpu addressing"),
    cl::init(256));

#define DEBUG_TYPE "ksan"

namespace {

#define DEBUG_PRINT 0

#if DEBUG_PRINT
static bool debug_printing = false;
//const char* debug_function = "init_worker_pool";
const char* debug_function = "foo";
#define debug_print(...) \
  if (debug_printing) { \
    __VA_ARGS__ \
  }
#else
#define debug_print(x)
#endif // DEBUG_PRINT

enum AccessType {
  NotAccessed = 0, // default
  Decoded = 1
};

/// HeapProtector: instrument memory accesses to prevent invalid heap accesses
struct HeapProtector : public FunctionPass {
public:
  explicit HeapProtector()
      : FunctionPass(ID) {
    initializeHeapProtectorPass(*PassRegistry::getPassRegistry());
  }
  
  StringRef getPassName() const override {
    return "HeapProtectorPass";
  }

  bool runOnFunction(Function &F) override;
  
  static char ID;  // Pass identification, replacement for typeid

private:
  Function *Func;
  Module *M;

  //DenseSet<pair<Value*, BasicBlock*>> DecodedInParent;

  /* edges from root/passthrough to passthrough/leaf */
  std::unordered_map<
    Value*,
    DenseSet<
      Use*         // this Use should be rewritten to point to the decoded value instead
    >
  > DecodedUsers;
  DenseSet<Value*> DecodedUseRoots;
  SmallVector<Use*, 8> FramePinArgs;

  bool needsDecoding(Use* AddressUse);
  void markOperandDecoded(Instruction *I, int Index);

  void findPtrSinksInCall(Instruction* I);
  void findPtrSinks(Instruction* I);

  Value *decodePtr(IRBuilder<> &IRB, Value *Ptr);
  Value *decodePtrAtSource(Value *Ptr);

  Value *UnderlyingHeapObject(Use *U,
        std::unordered_map<Value*, DenseSet<Use*>> *DecodedUsers,
        std::vector<Use*> *SourceChain);
};

} // anonymous namespace

char HeapProtector::ID = 0;
INITIALIZE_PASS_BEGIN(
    HeapProtector, "khp",
    "HeapProtector: temporal memory safety for kmalloc.", false,
    false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(
    HeapProtector, "khp",
    "HeapProtector: temporal memory safety for kmalloc.", false,
    false)

namespace llvm {
FunctionPass *createHeapProtectorFunctionPass() {
  return new HeapProtector();
}
}

Value *HeapProtector::UnderlyingHeapObject(Use *U,
        std::unordered_map<Value*, DenseSet<Use*>> *DecodedUsers,
        std::vector<Use*> *SourceChain) {
  while (1) {
    Value *V = U->get();
    if (DecodedUsers)
      (*DecodedUsers)[V].insert(U);
    if (SourceChain)
      SourceChain->push_back(U);

    Instruction *I = dyn_cast<Instruction>(V);
    if (!I)
      return V;

    U = nullptr;
    if (isa<BitCastInst>(I) || isa<AddrSpaceCastInst>(I) || isa<GetElementPtrInst>(I)) {
      U = &I->getOperandUse(0);
    }
    /* Don't step through number-to-pointer casts for now. */
    if (U == nullptr || !U->get()->getType()->isPointerTy())
      return V;
  }
}

/* Verify that @Address is pointer-typed, one of Argument and Instruction, and
 * not known to be a non-heap pointer.
 */
bool HeapProtector::needsDecoding(Use* AddressUse) {
  Value *Address = AddressUse->get();
  PointerType *AddressTy = dyn_cast<PointerType>(Address->getType());

  // We might pass scalar values into here (from CallInst handling) so the first
  // thing we need to check is that Address is actually a pointer.
  if (!AddressTy)
    return false;

  // Do not instrument acesses from different address spaces; we cannot deal
  // with them.
  if (AddressTy->getAddressSpace() != 0)
    return false;

  // Do not instrument accesses to function pointers, they should not be pointing
  // to the heap anyway
  if (AddressTy->getElementType()->isFunctionTy())
    return false;

  // Ignore swifterror addresses.
  // swifterror memory addresses are mem2reg promoted by instruction
  // selection. As such they cannot have regular uses like an instrumentation
  // function and it makes no sense to track them as memory.
  if (Address->isSwiftError())
    return false;

  Value* UnderlyingObject = UnderlyingHeapObject(AddressUse, nullptr, nullptr);

  /* Decoding can only be necessary for Arguments and Instructions. */
  if (isa<Argument>(UnderlyingObject))
    return true;

  Instruction *UnderlyingInsn = dyn_cast<Instruction>(UnderlyingObject);
  if (UnderlyingInsn == nullptr)
    return false;

  // Ignore addresses that are known to be pointers to stack buffers
  if (isa<AllocaInst>(UnderlyingInsn))
    return false;

  // TODO: need to think about what happens with objects aliased by virtue of
  // PtrToInt or BitCast.

  return true;
}

/*
 * Record the origin of operand @Index of @I in DecodedUsers and
 * DecodedUseRoots, unless we either have done so already or can prove that it
 * can't be an encoded pointer.
 */
void HeapProtector::markOperandDecoded(Instruction *I, int Index) {
  Use *AddressUse = &I->getOperandUse(Index);

  if (!needsDecoding(AddressUse))
    return;

  if (CallInst *CI = dyn_cast<CallInst>(I)) {
    /*
     * Any function call that gets a KHP-decoded pointer as argument definitely
     * must not be turned into a tail call.
     * (But this probably can't happen anyway, since we have a volatile store
     * afterwards.)
     */
    CI->setTailCallKind(llvm::CallInst::TCK_NoTail);
  }

  if (DecodedUsers[AddressUse->get()].count(AddressUse) == 1)
    return;
  Value *UnderlyingObject = UnderlyingHeapObject(AddressUse, &DecodedUsers, nullptr);
  DecodedUseRoots.insert(UnderlyingObject);
}

/*
 * special instructions:
 *  - Call
 */
static std::unordered_map<unsigned, std::vector<int>> insn_arg_info_by_opcode({
  { Instruction::AtomicCmpXchg, { Decoded, NotAccessed, NotAccessed } },
  { Instruction::AtomicRMW, { Decoded, NotAccessed } },
  { Instruction::Load, { Decoded } },
  { Instruction::Store, { NotAccessed, Decoded } },
});
static std::vector<int> &get_insn_arg_info(unsigned opcode) {
  static std::vector<int> empty_arg_info;

  auto arg_info_iter = insn_arg_info_by_opcode.find(opcode);
  if (arg_info_iter == insn_arg_info_by_opcode.end())
    return empty_arg_info;
  return arg_info_iter->second;
}

void HeapProtector::findPtrSinksInCall(Instruction *I) {
  CallInst *Call = cast<CallInst>(I);

  if (InlineAsm *Asm = dyn_cast<InlineAsm>(Call->getCalledOperand())) {
    // LLVM mixes calls and inline asm in its IR, but they're very different.
    InlineAsm::ConstraintInfoVector Constraints = Asm->ParseConstraints();
    int ArgIdx = -1;
    for (unsigned OperandNo=0; OperandNo<Constraints.size(); OperandNo++) {
      InlineAsm::ConstraintInfo &info = Constraints[OperandNo];

      // Direct outputs and clobbers don't come with arguments.
      bool IsWrite = info.Type == InlineAsm::ConstraintPrefix::isOutput;
      if (IsWrite && !info.isIndirect)
        continue;
      if (info.Type == InlineAsm::ConstraintPrefix::isClobber)
        continue;
      ArgIdx++;

      // We overload the meaning of indirect asm operands and assume that they
      // are always non-escaping, valid pointers.
      if (!info.isIndirect) continue;

      markOperandDecoded(Call, ArgIdx);
    }
    return;
  }

  if (Function* Callee = Call->getCalledFunction()) {
    if (Callee->isIntrinsic()) {
      switch (Callee->getIntrinsicID()) {
        case Intrinsic::memcpy:
        case Intrinsic::memcpy_element_unordered_atomic:
        case Intrinsic::memmove:
        case Intrinsic::memmove_element_unordered_atomic:
          markOperandDecoded(I, 0);
          markOperandDecoded(I, 1);
          break;
        case Intrinsic::memset:
        case Intrinsic::memset_element_unordered_atomic:
          markOperandDecoded(I, 0);
          break;
        default:
          break;
      }
    } else {
      auto Name = Callee->getName();
      if (Name == "memcpy" || Name == "memmove" || Name == "memcmp" || Name == "bcmp") {
        markOperandDecoded(I, 0);
        markOperandDecoded(I, 1);
      } else if (Name == "memset") {
        markOperandDecoded(I, 0);
      }
    }
  }

  for (unsigned ArgNo = 0; ArgNo < Call->getNumArgOperands(); ArgNo++) {
    if (!Call->isByValArgument(ArgNo))
      continue;
    markOperandDecoded(I, ArgNo);
  }
}

void HeapProtector::findPtrSinks(Instruction *I) {
  // TODO(encoded): how to avoid losing track of the origin of a value through
  // eg. bitcasts?

  // TODO(encoded): how to optimise for known-aliasing cases (like functions
  // which return a pointer into an argument)

  // TODO(decoded): think about other cases.

  // TODO(decoded, opt): reconsider handling of the cases which are handled by
  // llvm's intrinsics and optimisations (eg. memcpy, memset etc.)

  // TODO(decoded, opt): handle PHI nodes properly; propagate through them to
  // avoid double-decoding. This is super important for lifting decoding out of
  // loops.

  switch (I->getOpcode()) {
    case Instruction::Call:
      findPtrSinksInCall(I);
      break;
    default:
      std::vector<int> &iai = get_insn_arg_info(I->getOpcode());
      for (unsigned operand_idx = 0; operand_idx < iai.size(); operand_idx++) {
        if (operand_idx >= I->getNumOperands())
          break;
        if (iai[operand_idx] == Decoded)
          markOperandDecoded(I, operand_idx);
      }
      break;
  }
}

Value *HeapProtector::decodePtr(IRBuilder<> &IRB, Value *Ptr) {
  Type *Int8PtrTy = IRB.getInt8PtrTy();
  PointerType *Int8PtrPtrTy = Int8PtrTy->getPointerTo(0);

  // void *__khp_decode_ptr(void *encoded_ptr, void **stack_pin_slot)
  FunctionCallee DecodePtrCallee = M->getOrInsertFunction("__khp_decode_ptr", Int8PtrTy, Int8PtrTy, Int8PtrPtrTy);
  Value *DecodePtrFnVal = DecodePtrCallee.getCallee()->stripPointerCasts();
  Function *DecodePtrFn = cast<Function>(DecodePtrFnVal);

  /*
   * In order to reduce the impact of __khp_decode_ptr() on surrounding code and
   * make it easier to remove duplicate uses:
   *  - Use the preserve_most calling convention. __khp_decode_ptr() can be
   *    implemented with only two writable registers, and we want callers to
   *    actually be able to make use of the normally caller-saved registers.
   *    This is *NOT* preserve_all because that blows up for Linux kernel code.
   *    (`preserve_all` spills FPU registers, but the kernel doesn't
   *    context-switch them for kernel code.)
   *  - Pretend that we don't access any memory, in case it makes merging calls
   *    easier. This shouldn't have any effect on correctness.
   */
  DecodePtrFn->setDoesNotAccessMemory();
  DecodePtrFn->setDoesNotThrow();
  DecodePtrFn->setCallingConv(CallingConv::PreserveMost);

  Value *PtrCasted = IRB.CreatePointerCast(Ptr, Int8PtrTy);
  CallInst *DecodeCall = IRB.CreateCall(DecodePtrFn, {
    PtrCasted,
    ConstantPointerNull::get(Int8PtrPtrTy)
  });
  FramePinArgs.append({&DecodeCall->getOperandUse(1)});
  Value* NewPtr = IRB.CreatePointerCast(DecodeCall, Ptr->getType(), Ptr->getName()+"_decoded");

  return NewPtr;
}

Value *HeapProtector::decodePtrAtSource(Value *Ptr) {
  if (Instruction *I = dyn_cast<Instruction>(Ptr)) {
    if (isa<PHINode>(I)) {
      /* PHI nodes are in the BB prologue, we can't insert calls there */
      BasicBlock *BB = I->getParent();
      IRBuilder<> IRB(BB, BB->getFirstInsertionPt());
      return decodePtr(IRB, Ptr);
    } else {
      /* normal instruction, decode directly afterwards */
      IRBuilder<> IRB(I->getNextNode());
      return decodePtr(IRB, Ptr);
    }
  } else {
    /* function argument, decode on function entry */
    Argument *A = cast<Argument>(Ptr);
    BasicBlock *BB = &A->getParent()->getEntryBlock();
    IRBuilder<> IRB(BB, BB->getFirstInsertionPt());
    return decodePtr(IRB, Ptr);
  }
}

bool HeapProtector::runOnFunction(Function &F) {
  LLVMContext &C = F.getContext();
  Func = &F;

  DenseMap<Value*, Value*> DecodedOutputs;
  /*
   * WARNING: assumes that inputs are never remapped to other-typed nodes;
   * true for now, need to switch to Use* if that changes
   */
  DenseMap<Value*, Value*> DecodedInputs;

  DecodedUsers.clear();
  DecodedUseRoots.clear();
  FramePinArgs.clear();

#if DEBUG_PRINT
  if (F.getName().startswith(debug_function)) {
    errs() << "STARTING DEBUG PRINTS, FOUND " << F.getName() << "\n";
    debug_printing = true;
  }
  //debug_printing = true;
#endif

  M = F.getParent();

  for (auto &BB : F) {
    for (auto &I : BB) {
      debug_print(errs() << "   ### LOOKING AT: "; I.print(errs()); errs() << "\n";)
      findPtrSinks(&I);
    }
  }

  if (DecodedUseRoots.empty()) {
#if DEBUG_PRINT
    debug_printing = false;
#endif
    return false;
  }

  /* decode root values */
  for (Value *V: DecodedUseRoots) {
    DecodedOutputs[V] = decodePtrAtSource(V);
    assert(V->getType() == DecodedOutputs[V]->getType());
  }

  /* for non-root values, map encoded instructions to decoded ones */
  for (auto &iter2: DecodedUsers) {
    Value *V = iter2.first; /* root/passthrough */

    /* nothing to do if this is a root */
    if (DecodedOutputs[V])
      continue;
    /* V must be a passthrough */

    /* passthroughs are always instructions */
    Instruction *I = cast<Instruction>(V); /* passthrough/leaf */

    /* Passthrough instructions want substitution iff every use is decoded,
     * ***recursively for every passthrough***.
     * For now, just always duplicate passthrough instructions and let LLVM
     * sort out the dead code.
     */
    assert(!isa<LoadInst>(I));
    Instruction *NI = I->clone();
    NI->insertAfter(I);
    DecodedInputs[V] = NI;
    DecodedOutputs[V] = NI;
    assert(V->getType() == NI->getType());
  }

  /* fix up links between decoded values */
  for (auto &iter2: DecodedUsers) {
    Value *V = iter2.first; /* root/passthrough */
    Value *v_decoded = DecodedOutputs[V];
    assert(v_decoded->getType() == V->getType());
    auto &users = iter2.second;

    for (Use *U: users) {
      assert(U->get() == V);
      Value *decoded_user = DecodedInputs[U->getUser()]; /* passthrough/leaf */
      Use *u_decoded;
      if (decoded_user) {
        u_decoded = &cast<User>(decoded_user)->getOperandUse(U->getOperandNo()); /* decoded_user is passthrough */
      } else {
        u_decoded = U; /* decoded_user is leaf */
      }

      assert(u_decoded->get() == V);
      u_decoded->set(v_decoded);
    }
  }

  size_t pin_count = FramePinArgs.size();
  assert(pin_count != 0);
  IRBuilder<> IRB_entry(&Func->getEntryBlock(), Func->getEntryBlock().getFirstInsertionPt());

  /*
   * percpu_pin_head is a percpu variable; locate it in a target-specific way.
   */
  Constant *PercpuPinHead;
  if (Triple(M->getTargetTriple()).getArch() == Triple::ArchType::x86_64) {
    PercpuPinHead = ConstantExpr::getIntToPtr(
        IRB_entry.getInt32(PercpuPinHeadOffset),
        IRB_entry.getInt8PtrTy()->getPointerTo(PercpuPinHeadAS));
  } else {
    report_fatal_error("unsupported architecture");
  }

  /*
   * Allocate a structure like this on the stack:
   *     struct pins_frame {
   *       void *previous_frame;
   *       ptrsize_t pins_count;
   *       void *pins[pin_count];
   *     } __khp_frame_pin_area;
   */
  unsigned ptr_size = M->getDataLayout().getPointerTypeSize(IRB_entry.getInt8PtrTy());
  unsigned pins_size = pin_count * ptr_size;
  Type *AllocaTy = ArrayType::get(IRB_entry.getInt8PtrTy(), pin_count + 2);
  AllocaInst *FramePinArea = IRB_entry.CreateAlloca(AllocaTy, nullptr, "__khp_frame_pin_area");
  FramePinArea->setMetadata("no_stack_protector_needed", MDNode::get(C, {}));
  Value *PrevFramePtr = IRB_entry.CreateGEP(FramePinArea, {
    IRB_entry.getInt32(0),
    IRB_entry.getInt32(0)
  }, "prev_frame_ptr");
  Value *PinsCountPtr = IRB_entry.CreateGEP(FramePinArea, {
    IRB_entry.getInt32(0),
    IRB_entry.getInt32(1)
  }, "pins_count_ptr");

  /* memset(__khp_frame_pin_area.pins, 0, sizeof(__khp_frame_pin_area.pins)) */
  Value *PinAreaPins = IRB_entry.CreateGEP(FramePinArea, {
    IRB_entry.getInt32(0),
    IRB_entry.getInt32(2)
  }, "frame_pin_area_pins");
  IRB_entry.CreateMemSet(PinAreaPins, IRB_entry.getInt8(0), pins_size, MaybeAlign(None));
  /* pins_frame.previous_frame = percpu_pin_head */
  IRB_entry.CreateStore(
    IRB_entry.CreateLoad(
      IRB_entry.getInt8PtrTy(),
      PercpuPinHead,
      "old_percpu_pin_head"
    ),
    PrevFramePtr
  );
  /* pins_frame.pins_count = (void*){pin_count} */
  IRB_entry.CreateStore(
    ConstantExpr::getIntToPtr(
      IRB_entry.getInt32(pin_count),
      IRB_entry.getInt8PtrTy()
    ),
    PinsCountPtr
  );
  /*
   * percpu_pin_head = &pins_frame
   * (pins_frame must be fully initialized at this point, in case an interrupt
   * arrives right after this store)
   */
  IRB_entry.CreateStore(
    IRB_entry.CreatePointerCast(FramePinArea, IRB_entry.getInt8PtrTy()),
    PercpuPinHead,
    true /* must be volatile so that all preceding stores always happen */
  )->setAtomic(AtomicOrdering::Unordered);

  /* On exit, restore the old frame pin stack state. */
  for (auto &BB : F) {
    Instruction *Terminator = BB.getTerminator();
    if (!isa<ReturnInst>(Terminator))
      continue;
    IRBuilder<> IRB_exit(Terminator);

    /* percpu_pin_head = pins_frame.previous_frame */
    IRB_exit.CreateStore(
      IRB_exit.CreateLoad(
        IRB_exit.getInt8PtrTy(),
        PrevFramePtr,
        "old_percpu_pin_head"
      ),
      PercpuPinHead,
      true
    )->setAtomic(AtomicOrdering::Unordered);
  }

  /*
   * Turn the `__khp_decode_ptr({encoded_ptr}, NULL)` calls we created earlier
   * into `__khp_decode_ptr({encoded_ptr}, &__khp_frame_pin_area.pins[i])`.
   */
  unsigned FramePinIdx = 0;
  for (Use *FramePinArg: FramePinArgs) {
    Value *Slot = IRB_entry.CreateGEP(FramePinArea, {
      IRB_entry.getInt32(0),
      IRB_entry.getInt32(FramePinIdx + 2)
    }, "frame_pin_area_slot");
    FramePinArg->set(Slot);
    FramePinIdx++;
  }

#if DEBUG_PRINT
  debug_printing = false;
#endif

  return true;
}
