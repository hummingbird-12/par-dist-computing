SCAN_SRC  := scan
BLCK_SRC  := block
NBLCK_SRC := nblock

default:
	mpicc -Wall $(SCAN_SRC).c -o $(SCAN_SRC)
	mpicc -Wall $(BLCK_SRC).c -o $(BLCK_SRC)
	mpicc -Wall $(NBLCK_SRC).c -o $(NBLCK_SRC)

clean:
	rm -rf $(SCAN_SRC)
	rm -rf $(BLCK_SRC)
	rm -rf $(NBLCK_SRC)
