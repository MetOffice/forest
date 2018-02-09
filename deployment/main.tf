provider "aws" {
  region = "eu-west-2"
}

data "template_file" "bootstrap" {
  template = "${file("boot.sh.tlp")}"
}

resource "aws_instance" "forest_bokeh_server_dev21" {
  ami                   = "ami-e7d6c983"
  instance_type         = "t2.xlarge"	
  key_name              = "kubernetes.cluster.k8s.informaticslab.co.uk-be:87:08:3a:ea:a2:9e:7e:be:c1:97:2a:42:9b:8a:05"
  user_data             = "${data.template_file.bootstrap.rendered}"
  iam_instance_profile  = "seasia-bokeh-on-ec2"
  root_block_device {
       volume_size = 80
   }
  tags {
    Name = "forest_server_dev21",
    EndOfLife = "2018-09-30",
    OfficeHours = false,
    Project = "SEAsia",
    ServiceCode = "ZZTLAB",
    ServiceOwner = "aws@informaticslab.co.uk",
    Owner = "stephen.haddad"
  }
  security_groups        = ["default", "${aws_security_group.server.name}"]
}

resource "aws_security_group" "server" {
  name = "forest_server"
}


resource "aws_security_group_rule" "server" {
  description = "Allow web traffic to server"

  type        = "ingress"
  from_port   = 8888
  to_port     = 8888
  protocol    = "tcp"
  cidr_blocks = ["0.0.0.0/0"]
  security_group_id = "${aws_security_group.server.id}"

}

resource "aws_security_group_rule" "server_egress" {
  description = "Allow all egress"
  type        = "egress"
  from_port   = 0
  to_port     = 0
  protocol    = "-1"
  cidr_blocks = ["0.0.0.0/0"]
  security_group_id = "${aws_security_group.server.id}"
}




resource "aws_security_group_rule" "server_metoffice_ssh" {
  description = "Allow ssh from met office ip"
  type        = "ingress"
  from_port   = 22
  to_port     = 22
  protocol    = "tcp"
  cidr_blocks = ["151.170.0.0/16"]
  security_group_id = "${aws_security_group.server.id}"
}


resource "aws_security_group_rule" "server_gateway_ssh" {
  description = "Allow ssh from gateway ip"
  type        = "ingress"
  from_port   = 22
  to_port     = 22
  protocol    = "tcp"
  cidr_blocks = ["52.208.180.144/32"]
  security_group_id = "${aws_security_group.server.id}"
}

resource "aws_route53_record" "server_nice_url" {
  zone_id = "Z3USS9SVLB2LY1"
  name = "fdev21.informaticslab.co.uk"
  type = "A"
  ttl = "60"
  records = ["${aws_instance.forest_bokeh_server_dev21.public_ip}"]
}