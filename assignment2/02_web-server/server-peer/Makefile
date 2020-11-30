OUT := server.out

default:
	gcc -Wall server.c worker.c req_queue.c  -lpthread -o $(OUT)

clean:
	rm -rf $(OUT)
