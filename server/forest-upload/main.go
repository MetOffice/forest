package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"github.com/cheggaaa/pb/v3"
	"io"
	"io/ioutil"
	"log"
	"mime/multipart"
	"net/http"
	"os"
)

// Compile-time variable hidden from end-user
// use go build -ldflags "-X main.endpoint=$ENDPOINT"
var endpoint string

// Debug setting to help developers see response contents
var debug bool = false

type Namespace struct {
	APIKey    string
	fileNames []string
}

func parseArgs(argc []string) (Namespace, error) {
	flagSet := flag.NewFlagSet(argc[0], flag.ContinueOnError)
	APIKey, ok := os.LookupEnv("FOREST_API_KEY")
	if !ok {
		return Namespace{}, errors.New("FOREST_API_KEY environment variable not set")
	}
	err := flagSet.Parse(argc[1:])
	if err != nil {
		return Namespace{}, err
	}
	return Namespace{APIKey, flagSet.Args()}, nil
}

func Usage() {
	fmt.Printf("Usage: %s FILE [FILE ...]]\n", os.Args[0])
}

func main() {
	if endpoint == "" {
		log.Println("REST endpoint not specified during compilation")
		log.Fatalln("Contact an administrator")
	}
	args, err := parseArgs(os.Args)
	if err != nil {
		log.Fatal(err)
	}
	if len(args.fileNames) == 0 {
		Usage()
		fmt.Println("Too few arguments specified")
		return
	}
	for _, fileName := range args.fileNames {
		fmt.Printf("pre-sign URL: %s\n", fileName)
		url := endpoint + "?file=" + fileName
		content, err := apiKeyGet(url, args.APIKey)
		if err != nil {
			fmt.Printf("pre-signed URL generation failed: %s\n", fileName)
			log.Fatal(err)
		}
		signed, err := parseResponse(content)
		if err != nil {
			fmt.Printf("Could not parse response: %s\n", string(content))
			log.Fatal(err)
		}
		if debug {
			fmt.Printf("upload: %s to %s\n", fileName, signed.URL)
		} else {
			fmt.Printf("upload: %s to S3 bucket\n", fileName)
		}
		err = fileUpload(fileName, signed.URL, signed.Fields)
		if err != nil {
			log.Fatal(err)
		}
	}
}

func fileUpload(fileName string, url string, params map[string]string) error {
	// Read/write buffer to store file content
	rw := &bytes.Buffer{}

	// Multipart Writer
	writer := multipart.NewWriter(rw)

	// Add pre-signed form-fields
	for k, v := range params {
		writer.WriteField(k, v)
	}

	// Add file form-field at the end (AWS peculiarity)
	part, err := writer.CreateFormFile("file", fileName)
	if err != nil {
		return err
	}

	// Open file
	reader, err := os.Open(fileName)
	if err != nil {
		return err
	}
	defer reader.Close()

	// File size in bytes
	info, err := reader.Stat()
	if err != nil {
		return err
	}
	fileSize64 := info.Size()

	// Progress bar
	bar := pb.Full.Start(int(fileSize64))
	barReader := bar.NewProxyReader(reader)

	// Copy file using multipart.Writer
	if _, err = io.Copy(part, barReader); err != nil {
		return err
	}
	writer.Close()
	bar.Finish()

	// POST request and Println response (consider refactor)
	response, err := http.Post(url, writer.FormDataContentType(), rw)
	if err != nil {
		return err
	} else {
		body := &bytes.Buffer{}
		_, err := body.ReadFrom(response.Body)
		if err != nil {
			return err
		}
		response.Body.Close()
		if debug {
			fmt.Println(response.StatusCode)
			fmt.Println(response.Header)
			fmt.Println(body)
		}
	}
	return nil
}

func apiKeyGet(url, key string) ([]byte, error) {
	client := &http.Client{}
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return []byte{}, err
	}
	req.Header.Add("x-api-key", key)
	resp, err := client.Do(req)
	defer resp.Body.Close()
	content, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return []byte{}, err
	}
	return content, nil
}

type Response struct {
	URL    string            `json:"url"`
	Fields map[string]string `json:"fields"`
}

func parseResponse(content []byte) (Response, error) {
	resp := Response{}
	err := json.Unmarshal(content, &resp)
	if err != nil {
		return Response{}, err
	}
	return resp, nil
}
