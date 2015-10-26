import mnist_loader
import network
import time

# load data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

#net = network.Network([784, 30, 10])
#net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# training parameters
epochs = 50
mini_batch_size = 10

start_eta = 1.0
end_eta   = 100.0
delta_eta = 0.5

start_mid_layers_count = 0
end_mid_layers_count   = 10

start_neurons_count = 1
end_neurons_count   = 10000

# print them out
print "Parameters:"
print "\tepochs                 = %d" % epochs
print "\tmini_batch_size        = %d" % mini_batch_size

print "\tstart_eta              = %f" % start_eta
print "\tend_eta                = %f" % end_eta  
print "\tdelta_eta              = %f" % delta_eta

print "\tstart_mid_layers_count = %d" % start_mid_layers_count
print "\tend_mid_layers_count   = %d" % end_mid_layers_count  

print "\tstart_neurons_count    = %d" % start_neurons_count
print "\tend_neurons_count      = %d" % end_neurons_count  

# output file
# eta; layers; net_structure; epoch0; ...; epochN
header = "eta; layers; net_structure; "
for e in xrange(epochs):
    header += "epoch%d; " % e
output = open(time.strftime("%H.%M.%S_%d.%m%Y") + ".csv", "w")
output.write(header + "\n")

# training
etas = [start_eta + (delta_eta*i) for i in
        range(int((end_eta - start_eta)/delta_eta))]

try:
    for eta in etas:
        print "\neta: %f" % eta
        for layers in xrange(start_mid_layers_count, end_mid_layers_count):
            print "\t%r\tlayers: %d" % (time.ctime(), layers)
            neurons = [start_neurons_count for _ in xrange(layers)]
            while True:
                net_structure = [784, 10]
                net_structure[1:1] = neurons
                net = network.Network(net_structure)
                result = net.SGD(training_data, epochs, mini_batch_size, \
                                 eta, test_data)
                out = "%f; %d; %r; "
                for r in result:
                    out += "%d; " % r
                output.write(out + "\n")
                output.flush()
                if (layers == 0) or \
                   (layers > 0 and neurons[0] == end_neurons_count):
                    break
                for i in range(len(neurons) - 1, -1, -1):
                    if neurons[i] < end_neurons_count:
                        neurons[i] += 1
                        break
                    else:
                        neurons[i] = 0
except:
    pass
else:
    output.close()
