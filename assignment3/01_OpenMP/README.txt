Problem 1 - OpenMP Programming
------------------------------

* Instructions
1. Build using the `Makefile`
   $ make

2. Execute the program
   $ ./palindrome_linear THREAD_NUM INPUT_FILE OUTPUT_FILE
   $ ./palindrome_trie THREAD_NUM INPUT_FILE OUTPUT_FILE

   Example:
   $ ./palindrome_linear 256 words.txt result.txt
   $ ./palindrome_trie 512 words.txt result.txt

* Notes
- `palindrome_linear` uses linear search
- `palindrome_trie` uses search on trie data structure

* Developer
ID: 20161577
Name: Inho Kim
