 - Use the [AWS CLI to set up your credentals](http://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html#cli-quick-configuration) in `~/.aws`.

 - build: 
 ```
 docker build  -t bokeh .
 ```

 - run:
 ```
 docker run -i -t -p 8888:8888 --cap-add SYS_ADMIN --device /dev/fuse  -v ~/.aws:/root/.aws -v $(pwd)/documents:/opt/documents bokeh bash 
 ```

 - deploy:
 Requires `terraform` (`brew install terraform`)
 
 ``` 
 terraform init
 terraform apply
 ```
