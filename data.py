

import numpy as np
import cv2
import codecs
import cairocffi as cairo

# character classes and matching regex filter
regex = r'^[a-z ]+$'
alphabet = u'abcdefghijklmnopqrstuvwxyz '

class Data(object):
    """docstring for Da"""
    def __init__(self):
       # self.mnist = input_data.read_data_sets('data/')
        word_file = 'wordlists/clean_text.txt'
        self.f = codecs.open(word_file,mode='r',encoding='utf-8')
        self.lines = self.f.readlines()

    def next_batch(self, word_size, batch_size,epoch):
        imgs = []
        labels = []
        for i in range(batch_size):
            #ims, labs = self.mnist.train.next_batch(word_size)
            txt = self.get_word(i,batch_size,epoch)
            if(len(txt)==0):
            	print(" len value zero has come")
            	pass
            img = self.paint_text(txt,128,32)
            img = np.transpose(img)
            img = np.expand_dims(img,axis=3)
            imgs.append(img)
            labs = self.text_to_label(txt)
            labels.append(labs)
        labels = self.sparse_tuple_from(labels)
        return np.asarray(imgs), labels

    def sparse_tuple_from(self, sequences, dtype=np.int32):
        """Create a sparse representention of x.

        Args:
            sequences: a list of lists of type dtype where each element is a sequence
        Returns:
            A tuple with (indices, values, shape)
        """
        indices = []
        values = []        
        for n, seq in enumerate(sequences):
            indices.extend(zip([n] * len(seq), range(len(seq))))
            values.extend(seq)
     
        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=dtype)
        shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)     
        return indices, values, shape

    def decode_sparse_tensor(self, sparse_tensor):
        """Transform sparse to sequences ids."""
        decoded_indexes = list()
        current_i = 0
        current_seq = []
        for offset, i_and_index in enumerate(sparse_tensor[0]):
            i = i_and_index[0]
            if i != current_i:
                decoded_indexes.append(current_seq)
                current_i = i
                current_seq = list()
            current_seq.append(offset)
        decoded_indexes.append(current_seq)

        result = []
        for index in decoded_indexes:
            ids = [sparse_tensor[1][m] for m in index]
            text = ''.join(list(map(self.id2word, ids)))
            result.append(text)
        return result

    def hit(self, text1, text2):
        """Calculate accuracy of predictive text and target text."""
        res = []
        for idx, words1 in enumerate(text1):
            res.append(words1 == text2[idx])
        return np.mean(np.asarray(res))

    def id2word(self, idx):
        return str(idx)

    def text_to_label(self,text):
        """ converts text to """
        ret = []
        for char in text:
            ret.append(alphabet.find(char))
        return ret

    def labels_to_text(self,labels):
        ret = []
        for c in labels:
            if c == len(alphabet):
                ret.append("")
            else:
                ret.append(alphabet[c])

        return "".join(ret)

    def get_word(self,item_num,batch_size,epoch):
        epoch = epoch % 2000
        index = epoch*batch_size + item_num
        word = self.lines[index].rstrip()
        return word

    def paint_text(self,text, w, h):
        
        surface = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h)
        with cairo.Context(surface) as context:
            
            context.set_source_rgb(1, 1, 1)  # White
            context.paint()
            context.select_font_face('Courier',
                                     cairo.FONT_SLANT_NORMAL,
                                     cairo.FONT_WEIGHT_BOLD)
            context.set_font_size(15)
            box = context.text_extents(text)
            border_w_h = (4, 4)
            if box[2] > (w - 2 * border_w_h[1]) or box[3] > (h - 2 * border_w_h[0]):
                
                raise IOError(('Could not fit string into image.'
                           'Max char count is too large for given image width.'))

        # teach the RNN translational invariance by
        # fitting text box randomly on canvas, with some room to rotate
        max_shift_x = w - box[2] - border_w_h[0]
        max_shift_y = h - box[3] - border_w_h[1]
        top_left_x = np.random.randint(0, int(max_shift_x))
        top_left_y = h // 2

        context.move_to(top_left_x - int(box[0]), top_left_y - int(box[1]))
        context.set_source_rgb(0, 0, 0)
        context.show_text(text)

        buf = surface.get_data()
        a = np.frombuffer(buf, np.uint8)
        a.shape = (h, w, 4)
        a = a[:, :, 0]  # grab single channel
        a = a.astype(np.float32) / 255
        return a

    def num_classes(self):
        n_classes = len(alphabet)+1
        return(n_classes)


