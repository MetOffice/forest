BINARY=forest-upload
LDFLAGS="-X main.endpoint=${ENDPOINT}"

all: build build-linux
build:
	go build -ldflags ${LDFLAGS}
build-linux:
	GOOS=linux go build -o ${BINARY}-linux -ldflags ${LDFLAGS}
