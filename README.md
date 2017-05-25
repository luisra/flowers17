# Image-Based Flower Classification

## Synopsis

This project aims to evaluate the use of machine learning techniques to classify seventeen species of flowers. The study is focused on discussing how a linear SVC, in conjunction with a deep neural network (Inception-v3), is able to achieve a 95% classification accuracy.

## Code Example

In order to use Inception-v3 for feature extraction, we needed to create an instance of the trained model before initiating the extraction process.

Instance of the trained model:

```
def create_graph():

    with gfile.FastGFile(os.path.join(model_dir, "classify_image_graph_def.pb"), "rb") as f: 
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='') 
        
```

Feature extraction:

```
def extract_features(list_images): 

    nb_features = 2048
    features2 = np.empty((len(list_images), nb_features))
    labels = []
    
    create_graph() # Initiate instance of the trained model
    
    with tf.Session() as sess: # Retrieve penultimate layer
        penultimate_tensor = sess.graph.get_tensor_by_name('pool_3:0')

    for i in range(len(list_images)): # Feed image into layer and retrieve features and label  
        
        if (i%100 == 0):
            print("Processing %s..." % (list_images[i]))
        
        preds = sess.run(penultimate_tensor,
                           {'DecodeJpeg:0': images[i]})
        
        features2[i,:] = np.squeeze(preds)
        labels.append(re.split("_\d+",list_images[i].split("/")[1])[0])
    
    return features2, labels
```

Finally, we set up the desired model.

```
model2 = LinearSVC(C=1, loss='squared_hinge', penalty='l2',multi_class='ovr')
```

## Motivation

Before looking into features such as color, shape, and texture for classification purposes, we wanted to make use of an automated feature extraction method to establish a baseline. This would allow us to decide which approach to pursue further, automated vs. hands-on, moving forward.

## Installation

Provide code examples and explanations of how to get the project.

## API Reference

Depending on the size of the project, if it is small and simple enough the reference docs can be added to the README. For medium size to larger projects it is important to at least provide a link to where the API reference docs live.

## Tests

Describe and show how to run the tests with code examples.

## Contributors

Let people know how they can dive into the project, include important links to things like issue trackers, irc, twitter accounts if applicable.

## License

MIT
