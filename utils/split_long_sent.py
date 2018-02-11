import nltk
import cPickle


def main():
    root = '/scratch/wl1191/wsd2ed2/data/Semcor/docs2/senseTrain/'
    # root = '../../data/docs/senseTrain/'
    count_sub = dict()
    with open(root + 'unprocessed.txt', 'r') as flist:
        for line in flist:
            doc_name = line.rstrip('\r\n')
            with open(root + 'segments/' + doc_name + '.txt', 'r') as fin:
                doc = fin.read()[18:-20]

            sent = nltk.sent_tokenize(doc)
            fout = open(root + 'segments/' + doc_name + '_0.txt', 'w')
            fout.write('<DOC><BODY><TEXT>')
            index = 0

            for s in sent:
                s_split = s.split(' ')
                if len(s_split) < 100:
                    fout.write(' ' + s)
                else:
                    while len(s_split) > 100:
                        fout.write(' ' + ' '.join(s_split[:100]))
                        fout.write('</TEXT></BODY></DOC>')

                        s_split = s_split[100:]
                        fout.close()
                        index += 1
                        fout = open(root + 'segments/' + doc_name + '_' + str(index) + '.txt', 'w')
                        fout.write('<DOC><BODY><TEXT>')

                    fout.write(' ' + ' '.join(s_split[:100]))

            fout.write('</TEXT></BODY></DOC>')
            fout.close()

            count_sub[doc_name] = index

    print len(count_sub), 'files processed'
    cPickle.dump(count_sub, open(root + 'count_sub.pkl', 'w'))
    with open(root + 'sub.lst', 'w') as fout:
        for doc_name in count_sub:
            for i in range(count_sub[doc_name] + 1):
                fout.write(doc_name + '_' + str(i) + '.txt' + '\n')


if __name__ == '__main__':
    main()
