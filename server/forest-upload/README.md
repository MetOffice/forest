# FOREST collaboration file transfer tool

Transfer files to S3 using pre-signed URLs. This facility
enables collaborators to send small files to a dedicated
S3 bucket.

## Build process

The REST endpoint that generates pre-signed URLs can
be specified at compile time to reduce complexity
on the end user.

```bash
make build ENDPOINT=https://...amazonaws.com/lambda
```

**Note:** If the ENDPOINT is not specified at compile
time the application will fail to operate

## Usage

The REST endpoint is secured using API keys. Instructions
on how to use and store your API key should be made
available to you by an administrator. Once your
environment has been configured it is fairly simple
to use the tool.

```bash
# Upload multiple NetCDF files
forest-upload-${GOOS} *.nc
```

**Note:** Multi-part upload is not supported. The whole file is
read into memory prior to upload.

**Note:** FOREST_API_KEY should be in your environment when running the command
