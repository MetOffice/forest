package main

import (
	"os"
	"testing"
)

func TestParseResponse(t *testing.T) {
	t.Run("given empty JSON object returns error", func(t *testing.T) {
		_, err := parseResponse([]byte("{}"))
		if err == nil {
			t.Error("want error got nil")
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
