source = palindrome_linear.c palindrome_trie.c
app = palindrome_linear palindrome_trie
result = result.txt

all: $(app)

palindrome_%: palindrome_%.c
	gcc -g -Wall -fopenmp $< -o $@

clean:
	rm -f *.o $(result) $(app)
