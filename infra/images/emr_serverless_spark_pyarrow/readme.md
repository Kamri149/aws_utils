EMR Serverless custom image: Spark + pyarrow

Folder:
  infra/images/emr-serverless-spark-pyarrow

Base image:
  public.ecr.aws/emr-serverless/spark/emr-7.11.0:latest

Build/push is done via CodeBuild using buildspec.yml.
Default output:
  <account>.dkr.ecr.eu-west-2.amazonaws.com/trading-platform:emr-7.11.0-pyarrow-1
