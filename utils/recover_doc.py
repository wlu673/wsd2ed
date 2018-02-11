import sys


def main():
    root = '/scratch/wl1191/wsd2ed2/data/Semcor/'
    # root = 'data/'
    # datasets = ['senseTrain', 'senseValid', 'sense02', 'sense03', 'sense07']
    datasets = ['senseTrain']
    for dat in datasets:
        doc_dict = dict()
        doc_id_curr = ''
        doc_body = ''
        with open(root + dat + '.dat', 'r') as fin:
            index = 0
            for line in fin:
                print 'Processing line', index
                index += 1
                # doc_id = line.split('\t')[0].split('.')[0]
                doc_id = 'd00'
                if doc_id_curr != doc_id:
                    doc_id_curr = doc_id
                    doc_body = ''
                    doc_dict[doc_id_curr] = ''

                sent = line.split('\t')[2]
                if sent in doc_body:
                    continue
                sent = sent.split(' ')
                added = False
                # print '='*20 + '\n\n', doc_body, '\n\n' + '-'*20
                for i in range(len(sent)):
                    prefix = ' '.join(sent[:len(sent)-i])
                    # print prefix, '\n'
                    if doc_body.endswith(prefix):
                        doc_dict[doc_id_curr] += ' ' + ' '.join(sent[len(sent)-i:])
                        doc_body = ' '.join(sent)
                        added = True
                        break
                if not added:
                    doc_dict[doc_id_curr] += ' ' + ' '.join(sent)
                    doc_body = ' '.join(sent)
                # print '-'*20 + '\n\n', doc_body, '\n\n'
                sys.stdout.flush()
        # doc_dict[doc_id_curr] = doc_body

        print '\nWriting out\n'

        with open(root + 'docs2/' + dat + '.lst', 'w') as fnames:
            for doc_id in doc_dict:
                fnames.write(str(doc_id) + '.txt' + '\n')
                with open(root + 'docs2/' + dat + '/' + doc_id + '.txt', 'w') as fout:
                    fout.write('<DOC><BODY><TEXT>')
                    fout.write(doc_dict[doc_id])
                    fout.write('</TEXT></BODY></DOC>')


if __name__ == '__main__':
    main()
