# tensorflow_web_deploy
Deploy tensorflow inception model as web service running on gunicorn/flask instead of grpc.
Most of the stuffs are based on caffe web demo.

#how to make this work
Assume that you have installed tensorflow, flask, gunicorn. There may be other python module dependencies, pip install them.

1. git clone https://github.com/hetaoaoao/tensorflow_web_deploy
2. Put your inception model (named like model.ckpt-5000 or tensorflow-serving exported 'export' file) and its meta file to "data"
directory, rename them to model and model.meta.  Also modify the synset file accordingly.

  I already put a model.meta and a synset file there which is for the 5 flowers demo, trained model file is too large to upload to github.
  btw: the 5000 steps model get a 100% success rate:-)
  > Succesfully loaded model from /mnt/data1/tf/flowers5/models/model.ckpt-5000 at step=5000.
  > 2016-07-21 16:16:04.621380: starting evaluation on (validation).
  > 2016-07-21 16:16:09.724054: [20 batches out of 32] (125.4 examples/sec; 0.255sec/batch)
  > 2016-07-21 16:16:11.959930: precision @ 1 = 1.0000 recall @ 5 = 1.0000 [1024 examples]
3. Here we go, just 'sh run.sh', tail engine.log to check if something happens.
    If it works well, you should see something like.
  > INFO:PyClassification:classify_image cost 0.47 secs
  > 2016-07-21 17:30:37 [4374] [INFO] sample testing complete roses sunflowers dandelio
    
  when starting, it will try to classify the data/sample.jpg which is a rose picture.

4. Open your browser, navigate to "http://localhost:5001", the page should be simple enough to understand.
5. Use as webservice. Modify the interface for your need.
   curl --request POST --data-binary "@sample.jpg" http://localhost:5001
