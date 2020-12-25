SOURCE='palindrome.c'
EXE='palindrome'

INPUT='words.txt'
OUTPUT='result.txt'

gcc -g -Wall -fopenmp $SOURCE -o $EXE

i=1
while [ $i -le 8192 ]
do
    ./$EXE $i $INPUT $OUTPUT
    i=$((i * 2))
done

