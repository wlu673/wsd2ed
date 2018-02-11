def main():
    root = '/scratch/wl1191/wsd2ed2/data/'
    anns_dict, types = load_anns(root)
    print types

    out_names = {'train': 'eventTrain', 'valid': 'eventValid', "test": 'eventTest'}
    for dat in ['train', 'valid', 'test']:
        print 'Processing', dat
        with open(root + 'Semcor/' + out_names[dat] + '.entity.dat', 'w') as fout:
            with open(root + 'eventData/' + dat + '.txt', 'r') as fin:
                index = 0
                for line in fin:
                    line = line.rstrip('\r\n')
                    entries = line.split('\t')
                    if entries[0] not in anns_dict:
                        print entries[0], 'in', dat, 'not found in annotations'
                        exit(0)
                    fout.write(str(index) + '\t' + '\t'.join(entries[1:]) + '\t' + anns_dict[entries[0]] + '\n')
                    index += 1


def load_anns(root):
    anns_dict = dict()
    types = set()

    with open(root + 'triggerFeatureFile.WC_true.WED_true.fet.OfficialNN', 'r') as fin:
        for line in fin:
            if line.startswith('#'):
                continue
            entries = line.split('\t')

            if len(entries[-2]) == 0:
                anns_dict[entries[0]] = ''
                continue

            entities = entries[-2].split(' ')
            entities_dict = dict()
            for en in entities:
                en_info = en.split('#')
                types.add(en_info[2])
                ends = en_info[1].split('-')
                for index, pos in enumerate(range(int(ends[0]), int(ends[1]) + 1)):
                    entities_dict[pos] = en_info[2] + '-B' if index == 0 else en_info[2] + '-I'

            anns = ''
            for i in range(len(entries[-3].split(' '))):
                if i in entities_dict:
                    anns += entities_dict[i] + ';'
                else:
                    anns += 'OTHER' + ';'

            anns_dict[entries[0]] = anns[:-1]

    return anns_dict, types


if __name__ == '__main__':
    main()
