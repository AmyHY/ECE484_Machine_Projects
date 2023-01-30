/*
 * tab:2
 *
 * search_asm.S - Implementation of Assembly Recursive DFS
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for any purpose, without fee, and without written agreement is
 * hereby granted, provided that the above copyright notice and the following
 * two paragraphs appear in all copies of this software.
 *
 * IN NO EVENT SHALL THE AUTHOR OR THE UNIVERSITY OF ILLINOIS BE LIABLE TO
 * ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL
 * DAMAGES ARISING OUT  OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION,
 * EVEN IF THE AUTHOR AND/OR THE UNIVERSITY OF ILLINOIS HAS BEEN ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * THE AUTHOR AND THE UNIVERSITY OF ILLINOIS SPECIFICALLY DISCLAIM ANY
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE
 * PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND NEITHER THE AUTHOR NOR
 * THE UNIVERSITY OF ILLINOIS HAS ANY OBLIGATION TO PROVIDE MAINTENANCE,
 * SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS."
 *
 * Author:        Aamir Hasan
 * Version:       1
 * Creation Date: Fri Aug 30 2020
 * Filename:      search_asm.S
 * History:
 *    AS    1    Fri Aug 30 2020
 *        First written.
 */

.global binarySearch_asm

# Search ASM (Assembly)
# Searches for an element in a BST
# Declaration is in search.h
#
# Registers:
#   eax - Return Value
# 
binarySearch_asm:
  pushl %ebp
  movl %esp, %ebp

  #--- YOUR CODE HERE ---
  # EAX: arr
  # EBX: md
  # ECX: low
  # EDX: high
  # EDI: mid
  # ESI: arr[mid]
  # step 1: create the stack frame (already done above)
  PUSHL %EBX # protect callee-saved registers
  PUSHL %ESI 
  PUSHL %EDI
  #step 2: link to our input interface
  MOVL 8(%EBP), %EAX # arr <- M[EBP + 8]
  MOVL 12(%EBP), %EBX # md <- M[EBP + 12]
  MOVL 16(%EBP), %ECX # low <- M[EBP + 16]
  MOVL 20(%EBP), %EDX # high <- M[EBP + 20]
  # step 3: actual function logic
  XORL %EDI, %EDI # init mid = 0

  CMPL %ECX, %EDX # check high - low, jump to return -1 if < 0
  JL DOESNOTEXIST # return -1

  # mid <- low + high
  ADDL %ECX, %EDI
  ADDL %EDX, %EDI
  SAR %EDI # mid = mid / 2

  MOVL (%EAX, %EDI, 4), %ESI # ESI <- arr[mid]
  CMPL %EBX, %ESI # check arr[mid] - md, jump if not equal
  JE EQUAL # check md == arr[mid]
  JG SEARCHDOWN
  JL SEARCHUP

DOESNOTEXIST:
  MOVL $-1, %EAX # set return value to -1
  JMP TEARDOWN

EQUAL:
  MOVL $1, %EAX # set return value to 1
  JMP TEARDOWN

SEARCHDOWN:
  # prepare for recursive call
  # step 1: save caller-saved registers
  PUSHL %ECX
  PUSHFL
  # step 2: push arguments to stack
  ADDL $-1, %EDI
  PUSHL %EDI
  PUSHL %ECX
  PUSHL %EBX
  PUSHL %EAX
  # step 3: make the call
  CALL binarySearch_asm
  # step 4: pop arguments off the stack
  ADDL $16, %ESP
  # step 5: restore caller-saved registers
  POPFL
  POPL %ECX
  JMP TEARDOWN

SEARCHUP:
  # prepare for recursive call
  # step 1: save caller-saved registers
  PUSHL %ECX
  PUSHFL
  # step 2: push arguments to stack
  PUSHL %EDX
  ADDL $1, %EDI
  PUSHL %EDI
  PUSHL %EBX
  PUSHL %EAX
  # step 3: make the call
  CALL binarySearch_asm
  # step 4: pop arguments off the stack
  ADDL $16, %ESP
  # step 5: restore caller-saved registers
  POPFL
  POPL %ECX
  JMP TEARDOWN

  # step 4: tear down stack frame
TEARDOWN:
  POPL %EDI
  POPL %ESI
  POPL %EBX
  #----------------------
  leave
  ret
