provider "aws" {
  region = "eu-west-2"
}

data "template_file" "bootstrap" {
  template = "${file("boot.sh.tlp")}"
  vars {
    package = "${data.external.package.result["package"]}"
  }
}

data "external" "package" {
  program = ["bash", "package.sh"]
}

resource "aws_instance" "bokeh_server" {
  ami                   = "ami-e7d6c983"
  instance_type         = "t2.xlarge"	
  key_name              = "kubernetes.cluster.k8s.informaticslab.co.uk-be:87:08:3a:ea:a2:9e:7e:be:c1:97:2a:42:9b:8a:05"
  user_data             = "${data.template_file.bootstrap.rendered}"
  iam_instance_profile  = "seasia-bokeh-on-ec2"
  tags {
    Name = "se_asia_bokeh_demo",
    EndOfLife = "2018-02-10",
    OfficeHours = false,
    Project = "SEAsia",
    ServiceCode = "ZZTLAB",
    ServiceOwner = "aws@informaticslab.co.uk",
    Owner = "theo.mccaie"
  }
  security_groups        = ["default", "${aws_security_group.security_bokeh_server.name}", "${aws_security_group.security_metoffice.name}", "${aws_security_group.security_gateway.name}"]
}

resource "aws_security_group" "security_bokeh_server" {
  name = "security_bokeh_server"
  description = "Allow web traffic to bokeh_server"

  ingress {
      from_port = 8888
      to_port = 8888
      protocol = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
      from_port = 0
      to_port = 0
      protocol = "-1"
      cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "security_metoffice" {
  name = "security_metoffice"
  description = "Allow ssh from Met Office IPs"

  ingress {
      from_port = 22
      to_port = 22
      protocol = "tcp"
      cidr_blocks = ["151.170.0.0/16"]
  }
}


resource "aws_security_group" "security_gateway" {
  name = "security_gateway"
  description = "Allow ssh from gateway."

  ingress {
      from_port = 22
      to_port = 22
      protocol = "tcp"
      cidr_blocks = ["52.208.180.144/32"]
  }
}


