### CS231n Assignment3 - Training and Understanding Deep Networks

This final assignment of the course challenged me to explore how deep learning can be applied to problems beyond classification. I worked on generating captions for images, training generative models(GANs), and using self-supervised techniques to improve visual recognition with minimal labels.

RNN_Captioning.ipynb - Captioning Images Using Vanilla RNNs
I started by implementing a basic RNN model from scratch and used it to generate captions for images from the COCO dataset. The project deepened my understanding of sequential data processing and helped me get confortable with training loops for time-series inputs.

`Transformer_Captioning.ipynb` - Replacing RNNs with Transformers
Here, I reimplemented the image captioning pipeline using a Transformer model. This part helped me understand how self-attention and positional encoding work, and how they improve the ability to model long-range dependencies compared to RNNs.

`Generative_Adversarial_Networks.ipynb` - Training GANs from Scratch 
I built a GAN by defining both the generator and discriminator models, then trained it to generator image samples that resumble the training distribution. This section was challenging but rewarding. I learned about training stability, adversarial loss, and how GANs can be extended to help in semi-supervised learning tasks.

`Self_Supervised_Learning.ipynb` - Learning Visual Features Without Labels
I implemented a self-supervised learning technique that learns useful visual features by comparing different augmented views of the same image. This model was then fine-tuned for image classification using limited labeled data, highlighting the power of representation learning without explicit supervision.

`LSTM_Captioning.ipynb`
As an extra credit task, I also explored LSTM networks. Compared to vanilla RNNs, LSTMs were more effective in retaining information over longer sequences, which made a noticeable difference in the quality of image captions.

This assignment showed me how deep learning extends beyond classification into creative and unsupervised territory. Implementing each part from scratch solidified my understanding of both the math and intuition behind models like RNNs, Transformers, and GANs. The self-supervised learning section was especially exciting -- it feels like the future of scalable AI.

