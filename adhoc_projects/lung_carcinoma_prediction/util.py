import sys
import csv


def print_function(i, num_batches, acc_temp):
    sys.stdout.write('\r{0}'.format('Batches: ' + str(i+1) + '/' + str(num_batches) +\
                                    ' ' + '='*i + '>' + (num_batches-i-1) * ' ' + \
                                    str(round((i+1)*100/num_batches)) + '%; Accuracy: ' + \
                                   str(acc_temp)))
    sys.stdout.flush()
    pass


def predict(sess, x, keep_prob, pred, Xtest, output_file):
    feed_dict = {x:Xtest, keep_prob: 1.0}
    prediction = sess.run(pred, feed_dict=feed_dict)

    with open(output_file, "w") as file:
        writer = csv.writer(file, delimiter = ",")
        writer.writerow(["id","label"])
        for i in range(len(prediction)):
            writer.writerow([str(i), str(prediction[i])])

    print("Output prediction: {0}". format(output_file))