package main

import (
	"encoding/json"
	"testing"
)

func TestParseResponse(t *testing.T) {
	t.Run("given empty JSON object returns empty struct", func(t *testing.T) {
		resp, _ := parseResponse([]byte("{}"))
		if resp.URL != "" {
			t.Error("Should get empty URL")
		}
	})

	t.Run("urls/fields", func(t *testing.T) {
		url := "s3-bucket.url"
		fields := map[string]string{"key": "value"}
		resp := Response{url, fields}
		b, err := json.Marshal(resp)
		if err != nil {
			t.Error(err)
		}
		signed, err := parseResponse(b)
		if err != nil {
			t.Error(err)
		}
		if signed.URL != url {
			t.Errorf("want %s got %s", url, signed.URL)
		}
		if signed.Fields["key"] != fields["key"] {
			t.Errorf("want %s got %s", fields, signed.Fields)
		}
	})
}

func TestParseArgs(t *testing.T) {
	t.Run("file(s)", func(t *testing.T) {
		args, err := parseArgs([]string{"cmd", "file.nc"})
		if err != nil {
			t.Errorf("got error: %s", err)
		}
		if len(args.fileNames) != 1 {
			t.Errorf("got: %s", args.fileNames)
		}
	})

	t.Run("-version", func(t *testing.T) {
		args, err := parseArgs([]string{"cmd", "-version"})
		if err != nil {
			t.Errorf("got error: %s", err)
		}
		if args.version != true {
			t.Errorf("want %t got %t", true, args.version)
		}
	})
}
