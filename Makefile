# "make" or "make lunar" - build lunar executable
# "make run"             - build lunar executable and run it
# "make test"            - build and run unit tests
# "make clean"           - delete executable and test output files
# "make build-docker"    - build oldmankris/lunar Docker image
# "make run-docker"      - run oldmankris/lunar Docker image
# "make push-docker"     - push oldmankris/lunar Docker image to docker.io

CFLAGS:=-O3 -Wall
LDLIBS:=-lm
DIFF:=diff
DOCKER:=docker

DOCKER_NAME:=oldmankris/lunar
DOCKER_TAG:=latest

DOCKER_BUILD_FLAGS:=--squash --no-cache

lunar: lunar.c

run: lunar
	./lunar
.PHONY: run

test: test_success test_failure test_good
.PHONY: test

test_good: lunar
	./lunar --echo <test/good_input.txt >good_output.txt
	$(DIFF) test/good_output_expected.txt good_output.txt
.PHONY: test_good

test_success: lunar
	./lunar --echo <test/success_input.txt >success_output.txt
	$(DIFF) test/success_output_expected.txt success_output.txt
.PHONY: test_success

test_failure: lunar
	./lunar --echo <test/failure_input.txt >failure_output.txt
	$(DIFF) test/failure_output_expected.txt failure_output.txt
.PHONY: test_failure

build-docker:
	$(DOCKER) build $(DOCKER_BUILD_FLAGS) -t $(DOCKER_NAME):$(DOCKER_TAG) .
.PHONY: build-docker

run-docker:
	$(DOCKER) run -it --rm $(DOCKER_NAME):$(DOCKER_TAG)
.PHONY: run-docker

push-docker:
	$(DOCKER) push $(DOCKER_NAME):$(DOCKER_TAG)
.PHONY: push-docker

clean:
	- $(RM) lunar
	- $(RM) success_output.txt
	- $(RM) failure_output.txt
	- $(RM) good_output.txt
.PHONY: clean
