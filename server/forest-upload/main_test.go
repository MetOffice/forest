package main

import (
	"encoding/json"
	"os"
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
	t.Run("given environment variable", func(t *testing.T) {
		key := "ABC"
		os.Setenv("FOREST_API_KEY", key)
		got, err := parseArgs([]string{"file.nc"})
		if err != nil {
			t.Errorf("got error: %s", err)
		}
		if got.APIKey != key {
			t.Errorf("want %s got %s", key, got.APIKey)
		}
	})

	t.Run("without api key environment variable", func(t *testing.T) {
		os.Unsetenv("FOREST_API_KEY")
		_, err := parseArgs([]string{"file.nc"})
		if err == nil {
			t.Error("expect error to be raised")
		}
	})

}
