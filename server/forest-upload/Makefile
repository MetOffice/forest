BINARY=forest-upload
VERSION=0.1.1
LDFLAGS="-X main.endpoint=${ENDPOINT} -X main.version=${VERSION}"

all: build build-linux
build:
	go build -ldflags ${LDFLAGS}
build-linux:
	GOOS=linux go build -o ${BINARY}-linux -ldflags ${LDFLAGS}
