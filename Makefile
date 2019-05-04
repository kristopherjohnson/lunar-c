CFLAGS:=-O3
DIFF:=diff

lunar: lunar.c

run: lunar
	./lunar
.PHONY: run

test: test_success test_failure
.PHONY: test

test_success: lunar
	./lunar --echo <test/success_input.txt >success_output.txt
	$(DIFF) test/success_output_expected.txt success_output.txt
.PHONY: test_success

test_failure: lunar
	./lunar --echo <test/failure_input.txt >failure_output.txt
	$(DIFF) test/failure_output_expected.txt failure_output.txt
.PHONY: test_failure

clean:
	- $(RM) lunar
	- $(RM) success_output.txt
	- $(RM) failure_output.txt
.PHONY: clean
