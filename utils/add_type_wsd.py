import sys
import cPickle
import xml.etree.ElementTree as ET


def main():
    abbrev = {'LOC': 'LOCATION', 'WEA': 'WEA', 'GPE': 'GPE', 'PER': 'PERSON', 'FAC': 'FACILITY', 'ORG': 'ORGANIZATION',
              'VEH': 'VEH'}
    root = '/scratch/wl1191/wsd2ed2/data/Semcor/'
    # root = '../../data/'
    datasets = ['senseTrain']
    # datasets = ['senseValid', 'sense02', 'sense03', 'sense07']

    for dat in datasets:
        if dat == 'senseTrain':
            process_train_data(root, abbrev)
        else:
            process_data(root, dat, abbrev)

    # doc_ids = {'senseTrain': cPickle.load(open(root + 'docs/train_doc_id.pkl', 'r')),
    #            'sense02': ['d00', 'd01', 'd02'],
    #            'sense03': ['d000', 'd001', 'd002'],
    #            'sense07': ['d00', 'd01', 'd02'],
    #            'senseValid': ['d000', 'd001', 'd002']}
    # types = set()
    # for dat in datasets:
    #     for doc_id in doc_ids[dat]:
    #         doc_tree = ET.parse(root + 'docs/' + dat + 'Out/' + doc_id + '.txt.apf.xml')
    #         entities = doc_tree.getroot()[0].findall('entity')
    #         for en in entities:
    #             types.add(en.attrib['TYPE'])
    # print types


def process_data(root, dat, abbrev):
    doc_id = ''
    doc_body = ''
    anns = dict()

    with open(root + dat + '.entity.dat', 'w') as fout:
        with open(root + dat + '.dat', 'r') as fin:
            for line in fin:
                line = line.rstrip('\r\n')
                entries = line.split('\t')
                if doc_id != entries[1]:
                    doc_id = entries[1]
                    doc_body, anns = load_doc(doc_id, dat, root, abbrev)

                sent = entries[2]
                offset = doc_body.index(sent)
                entities = ''
                for t in sent.split(' '):
                    if offset in anns:
                        entities += anns[offset] + ';'
                    else:
                        entities += 'OTH;'
                    offset += len(t) + 1
                fout.write(line + '\t' + entities[:-1] + '\n')


def process_train_data(root, abbrev):
    doc_list = cPickle.load(open(root + 'docs/train_doc_id.pkl', 'r'))
    curr_index = 0
    doc_body, anns = load_doc(doc_list[curr_index], 'senseTrain', root, abbrev)
    doc_body_buffer = [doc_body]
    anns_buffer = [anns]
    offset_buffer = [0]
    count = 0
    with open(root + 'senseTrain.entity.dat', 'w') as fout:
        with open(root + 'senseTrain.dat', 'r') as fin:
            for line in fin:
                line = line.rstrip('\r\n')
                entries = line.split('\t')
                sent = entries[2]

                try:
                    while sent not in ''.join(doc_body_buffer):
                        if len(doc_body_buffer) > 2 and sum([len(d) for d in doc_body_buffer[1:]]) > len(sent):
                            doc_body_buffer = doc_body_buffer[1:]
                            anns_buffer = anns_buffer[1:]
                            offset_buffer = [o - offset_buffer[1] for o in offset_buffer[1:]]

                        curr_index = (curr_index + 1) % len(doc_list)
                        offset_buffer += [offset_buffer[-1] + len(doc_body_buffer[-1])]
                        doc_body, anns = load_doc(doc_list[curr_index], 'senseTrain', root, abbrev)
                        doc_body_buffer += [doc_body]
                        anns_buffer += [anns]
                        # print 'Current index', curr_index

                except KeyboardInterrupt:
                    print '\n', sent, '\n'
                    print doc_list[curr_index - 2: curr_index + 1]
                    exit(0)

                try:
                    offset0 = ''.join(doc_body_buffer).index(sent)
                except ValueError:
                    print sent, '\n'
                    print doc_list[curr_index-2: curr_index+1]
                    exit(0)

                offset = offset0
                entities = ''
                for t in sent.split(' '):
                    matched = False
                    for i, a in enumerate(anns_buffer):
                        o = offset - offset_buffer[i]
                        if o in a:
                            entities += a[o] + ';'
                            matched = True
                            break
                    if not matched:
                        entities += 'OTH;'
                    offset += len(t) + 1

                for i, o in enumerate(offset_buffer[1:]):
                    if offset0 < o:
                        doc_body_buffer = doc_body_buffer[i:]
                        anns_buffer = anns_buffer[i:]
                        offset_buffer = [0]
                        for d in doc_body_buffer[:-1]:
                            offset_buffer += [offset_buffer[-1] + len(d)]
                        break

                fout.write(line + '\t' + entities[:-1] + '\n')

                count += 1
                if count % 50 == 0:
                    print count
                    sys.stdout.flush()


def load_doc(doc_id, dat, root, abbrev):
    with open(root + 'docs/' + dat + '/' + doc_id + '.txt', 'r') as fdoc:
        contents = fdoc.read()
        start = contents.index('<TEXT>') + 6
        end = contents.index('</TEXT>')
        doc_body = contents[start:end]

    doc_tree = ET.parse(root + 'docs/' + dat + 'Out/' + doc_id + '.txt.apf.xml')
    entities = doc_tree.getroot()[0].findall('entity')
    anns = dict()
    for en in entities:
        seq = en.find('entity_mention').find('extent').find('charseq')
        # print en.attrib['TYPE'], en.attrib['CLASS'], seq.attrib['START'], seq.attrib['END'], seq.text
        start, end = int(seq.attrib['START']), int(seq.attrib['END']) + 1
        if doc_body[start:end] != seq.text:
            print 'Annotation does not match contents in doc', doc_id
            print 'In annotation:', seq.text
            print 'In doc:', doc_body[start:end]
            print 'Start:', start
            print 'End::', end
            exit(0)
        tokens = seq.text.split(' ')
        offset = start
        for index, t in enumerate(tokens):
            anns[offset] = abbrev[en.attrib['TYPE']]
            anns[offset] += '-B' if index == 0 else '-I'
            offset += len(t) + 1

    return doc_body, anns


if __name__ == '__main__':
    main()
