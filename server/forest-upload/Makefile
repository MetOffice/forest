BINARY=forest-upload

all: build build-linux
build:
	go build -ldflags "-X main.endpoint=${ENDPOINT}"
build-linux:
	GOOS=linux go build -o ${BINARY}-linux -ldflags "-X main.endpoint=${ENDPOINT}"
