import matplotlib.pyplot as plt


plt.figure(1)

# max silence
plt.subplot(221)
y = [0.24312702,0.257319039,0.2732,0.27538156,0.298797538,0.26410641,0.252768894,0.251772083,0.252510629,0.25089521]
x = [3,5,7,9,11,13,15,17,19,21]
plt.plot(x, y, "o-")
plt.ylabel('Recall')
plt.title('Max Silence Length')
plt.grid(True)


# overlap
plt.subplot(222)
y = [0.274152605,0.281926658,0.289390208,0.302277312,0.307062001,0.307062001,0.307062001,0.307062001,0.307062001,0.307062001]
x = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
plt.plot(x, y, "o-")
plt.ylabel('Recall')
plt.title('Minimum Overlap with Existing Concepts')
plt.grid(True)


# symmetric log
plt.subplot(223)
y = [0.287198292,0.287198292,0.287198292,0.291982981,0.302277312,0.307062001,0.307062001,0.293363856,0.288579167,0.285824346]
x = [10,11,12,13,14,15,16,17,18,19]
plt.plot(x, y, "o-")
plt.ylabel('Recall')
plt.title('Number of Neighborrs')
plt.grid(True)

# logit
plt.subplot(224)
y = [0.2177,0.2301,0.2485,0.2600,0.2681,0.273586235,0.283718518,0.298797538,0.307062001,0.306899952]
x = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
plt.plot(x, y, "o-")
plt.ylabel('Recall')
plt.title('Lambda')
plt.grid(True)

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

plt.show()
