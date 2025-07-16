

```
(env) cccimac@cccimacdeiMac 03-pydasm_arg % ./test.sh
  0           RESUME                   0

  1           LOAD_CONST               0 (<code object add at 0x100faf9f0, file "test/example.py", line 1>)
              MAKE_FUNCTION
              STORE_NAME               0 (add)

  4           LOAD_NAME                1 (print)
              PUSH_NULL
              LOAD_CONST               1 ('add(2,3)=')
              LOAD_NAME                0 (add)
              PUSH_NULL
              LOAD_CONST               2 (2)
              LOAD_CONST               3 (3)
              CALL                     2
              CALL                     2
              POP_TOP

  6           LOAD_CONST               4 (<code object foo at 0x100fafd70, file "test/example.py", line 6>)
              MAKE_FUNCTION
              STORE_NAME               2 (foo)

  9           LOAD_NAME                1 (print)
              PUSH_NULL
              LOAD_CONST               5 ('foo(1,2,3,4)=')
              LOAD_NAME                2 (foo)
              PUSH_NULL
              LOAD_CONST               6 (1)
              LOAD_CONST               2 (2)
              LOAD_CONST               3 (3)
              LOAD_CONST               7 (4)
              CALL                     4
              CALL                     2
              POP_TOP
              RETURN_CONST             8 (None)

Disassembly of <code object add at 0x100faf9f0, file "test/example.py", line 1>:
  1           RESUME                   0

  2           LOAD_FAST_LOAD_FAST      1 (a, b)
              BINARY_OP                0 (+)
              RETURN_VALUE

Disassembly of <code object foo at 0x100fafd70, file "test/example.py", line 6>:
  6           RESUME                   0

  7           LOAD_FAST_LOAD_FAST      1 (a, b)
              BINARY_OP                0 (+)
              LOAD_FAST                2 (c)
              BINARY_OP                0 (+)
              LOAD_FAST                3 (d)
              BINARY_OP                0 (+)
              RETURN_VALUE
  0           RESUME                   0

  1           LOAD_CONST               0 (<code object count_up_to at 0x1048829a0, file "test/yield1.py", line 1>)
              MAKE_FUNCTION
              STORE_NAME               0 (count_up_to)

  8           LOAD_NAME                0 (count_up_to)
              PUSH_NULL
              LOAD_CONST               1 (5)
              CALL                     1
              STORE_NAME               1 (counter)

 10           LOAD_NAME                1 (counter)
              GET_ITER
      L1:     FOR_ITER                11 (to L2)
              STORE_NAME               2 (number)

 11           LOAD_NAME                3 (print)
              PUSH_NULL
              LOAD_NAME                2 (number)
              CALL                     1
              POP_TOP
              JUMP_BACKWARD           13 (to L1)

 10   L2:     END_FOR
              POP_TOP
              RETURN_CONST             2 (None)

Disassembly of <code object count_up_to at 0x1048829a0, file "test/yield1.py", line 1>:
   1           RETURN_GENERATOR
               POP_TOP
       L1:     RESUME                   0

   2           LOAD_CONST               1 (1)
               STORE_FAST               1 (count)

   3           LOAD_FAST_LOAD_FAST     16 (count, max)
               COMPARE_OP              58 (bool(<=))
               POP_JUMP_IF_FALSE       17 (to L5)

   4   L2:     LOAD_FAST                1 (count)
               YIELD_VALUE              0
               RESUME                   5
               POP_TOP

   5           LOAD_FAST                1 (count)
               LOAD_CONST               1 (1)
               BINARY_OP               13 (+=)
               STORE_FAST               1 (count)

   3           LOAD_FAST_LOAD_FAST     16 (count, max)
               COMPARE_OP              58 (bool(<=))
               POP_JUMP_IF_FALSE        2 (to L4)
       L3:     JUMP_BACKWARD           16 (to L2)
       L4:     RETURN_CONST             0 (None)
       L5:     RETURN_CONST             0 (None)

  --   L6:     CALL_INTRINSIC_1         3 (INTRINSIC_STOPITERATION_ERROR)
               RERAISE                  1
ExceptionTable:
  L1 to L3 -> L6 [0] lasti
  L4 to L6 -> L6 [0] lasti
```