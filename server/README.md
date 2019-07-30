# FOREST continuous deployment pipeline

The scripts contained in this directory are used by AWS
CodeDeploy and CodePipeline services to deploy FOREST to
https://forest.informaticslab.co.uk. The `appspec.yml`
contained in the top level directory of the repository
tells CodeDeploy which scripts to run at which
point in the deployment life cycle.

## launch-config.sh

This script is not run directly but instead is manually copied and pasted
into the `user data` section of a new launch configuration. Launch
configuration user data sections are immutable after creation, so this
is a very convenient piece of text to keep around.

## appspec.yml

There are is one gotcha with this file, in the `files`
section, `source` refers to the repository and `destination`
refers to the directory on the EC2 instance that the
CodeDeploy agent uses to clone the repository

```markdown
files:
    source: /
    destination: /path/on/ec2/instance
```

There is very little documentation elsewhere on
the internet at present to explain the true meaning and
usage of that section

## Health check error code

As bokeh apps return a `302` redirect code instead of `200` on navigation to `/`, the EC2
application load balancer target group health check has been modified
to expect a `302` to signify a healthy FOREST instance
