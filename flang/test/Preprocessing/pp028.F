* fixed-form clipping done before KWM expansion on source line
      integer, parameter :: KW = 777
#define KWM 666
      integer :: res
* 'M' is in column 73
*        1         2         3         4         5         6         7
*234567890123456789012345678901234567890123456789012345678901234567890123
      res =                                                           KWM
      if (res .eq. 777) then
        print *, 'pp028.F pass'
      else
        print *, 'pp028.F FAIL: ', res
      end if
      end
