import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

print(mnist.train.images.shape,mnist.train.labels.shape)
print(mnist.validation.images.shape,mnist.validation.labels.shape)
print(mnist.test.images.shape,mnist.test.labels.shape)

#每个批次的大小
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size
n_test_batch = mnist.train.num_examples // batch_size

n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_output = 10

weight_decay=0.1
l2_reg=tf.contrib.layers.l2_regularizer(weight_decay)

x = tf.placeholder(tf.float32,[None,n_input])
y = tf.placeholder(tf.float32,[None,n_output])

def multilayer(X,weights,biases):
    layer_1 = tf.add(tf.matmul(X,weights['h1']),biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    out_layer = tf.matmul(layer_1,weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1],stddev=0.01)),
'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
'out': tf.Variable(tf.random_normal([n_hidden_2, n_output],stddev=0.01)),
'out1': tf.Variable(tf.random_normal([n_input,n_output],stddev=0.01))
}
biases = {
'b1': tf.Variable(tf.random_normal([n_hidden_1])),
'b2': tf.Variable(tf.random_normal([n_hidden_2])),
'out': tf.Variable(tf.random_normal([n_output],stddev=0.01))
}
y_ = tf.nn.softmax(multilayer(x,weights,biases))
loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_,labels=y)
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
#结果放在一个布尔列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))#argmax返回一维张量中最大的值所在的位置,1代表一行中的最大位置
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
accu = []
for epoch in range(20):
    for batch in range(n_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        train_step.run({x:batch_xs,y:batch_ys})
    acc = 0
    for batch in range(n_test_batch):
        batch_xs, batch_ys = mnist.test.next_batch(batch_size)
        #acc += accuracy.run({x:batch_xs,y:batch_ys})
        acc+=sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys})
    print("Iter " + str(epoch+1) + ",Testing Accuracy " + str(acc / (batch + 1)))
    accu.append(acc/(batch+1))

plt.plot(range(1,21),accu,color = 'b')
plt.show()

