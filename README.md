# Image-Based Flower Classification

Multiclass classifier, based on the [Flowers 17]( http://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html) Dataset. The data was obtained from the University of Oxford’s Department of Engineering Science.

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
    features = np.empty((len(list_images), nb_features))
    labels = []
    
    create_graph() # Initiate instance of the trained model
    
    with tf.Session() as sess: # Retrieve penultimate layer
        penultimate_tensor = sess.graph.get_tensor_by_name('pool_3:0')

    for i in range(len(list_images)): # Feed image into layer and retrieve features and label  
        
        if (i%100 == 0):
            print("Processing %s..." % (list_images[i]))
        
        preds = sess.run(penultimate_tensor,
                           {'DecodeJpeg:0': images[i]})
        
        features[i,:] = np.squeeze(preds)
        labels.append(re.split("_\d+",list_images[i].split("/")[1])[0])
    
    return features, labels
```

Finally, we set up the desired model.

```
model = LinearSVC(C=1, loss='squared_hinge', penalty='l2',multi_class='ovr')
```

## Motivation

Before looking into features such as color, shape, and texture for classification purposes, we wanted to make use of an automated feature extraction method to establish a baseline. This would allow us to decide which approach to pursue further, automated vs. hands-on, moving forward.

## Installation

Both the flowers17.py script and flowers17.ipynb notebook perform all aspects of this implementation.

## Tests

After feature extraction, we tested the performance of the linear SVC model.

Ten-Fold CV Accuracy Score:
```
print("Linear SVC Accuracy (Ten-Fold CV):", cross_val_score(model, features, labels, cv=10).mean(), "\n")
```

Holdout Set Accuracy Score (70/30):
```
Xtrain, Xtest, ytrain, ytest = train_test_split(features, labels,
                                                random_state = 7,
                                                test_size = 0.3
                                                )
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

print("Linear SVC Accuracy (Holdout Set):", accuracy_score(ytest, ypred), "\n")
```

Classification Report:
```
print("Linear SVC Classification Report:", "\n")
print(classification_report(ytest, model.predict(Xtest), 
                            target_names = classes))
```

Heatmap:
```
plt.figure(figsize=(8, 8))
mat = confusion_matrix(ytest, ypred)
ax = sns.heatmap(mat.T, square = True, annot = True, fmt='d', cbar=False,
            xticklabels = c, yticklabels= c)
plt.xlabel('true label')
plt.ylabel('pred label')
plt.title('Linear SVC Heatmap')
plt.show();
```

## References

[1] Dürr, Oliver. (2016) Deep learning for lazybones. https://oduerr.github.io/blog/2016/04/06/Deep-Learning_for_lazybones

## License

MIT License
