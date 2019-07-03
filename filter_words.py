import codecs
import cairocffi as cairo
w = 128
h = 28

file = 'words_alpha.txt'
out_file = 'clean_text.txt'
f2 = open(out_file, "a")

surface = cairo.ImageSurface(cairo.FORMAT_RGB24, 128, 28)
context = cairo.Context(surface)
context.set_source_rgb(1, 1, 1)  # White
context.paint()
context.select_font_face('Courier',
                                     cairo.FONT_SLANT_NORMAL,
                                     cairo.FONT_WEIGHT_BOLD)
context.set_font_size(20)
border_w_h = (4, 4)


f = codecs.open(file,mode='r',encoding='utf-8')

lines = f.readlines()
for line in lines:
        word = line.rstrip()
        box = context.text_extents(word)
        if box[2] > (w - 2 * border_w_h[1]) or box[3] > (h - 2 * border_w_h[0]):
                pass
        else:
                f2.write(line)




f.close()
f2.close()
#context.restore()

        





