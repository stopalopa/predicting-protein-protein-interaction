import random

positive_protein_pairs = {}
negative_protein_pairs = {}
all_protein_pairs = []
all_protein_pairs_rvs = []
protein_pair_labels = {}
proteins = {}
count = 0
protein = ""
random.seed(1)

with open("pos_data_test.txt") as input1:
    for line in input1:
        # check if line starts with number:
        if line[0].isdigit():
            line = line.strip('\n')
            line = line.replace('  ', ' ')
            pair = line.split(' ')            
            protein_pair_labels[pair[1] + "-" + pair[2]] = 1
            all_protein_pairs.append(pair[1] + "-" + pair[2])
            all_protein_pairs_rvs.append(pair[2] + "-" + pair[1])
             
        elif line[0] == '>':
            line = line.strip('\n')
            protein = line[1:]
        else:
            line = line.strip('\n')
            proteins[protein] = line
input1.close()
with open("neg_data_test.txt") as input2:
    for line in input2:
        # check if line starts with number:
        if line[0].isdigit():
            line = line.strip('\n')
            line = line.replace('  ', ' ')
            pair = line.split(' ')
                       
            protein_pair_labels[pair[1] + "-" + pair[2]] = 0
            all_protein_pairs.append(pair[1] + "-" + pair[2])
            all_protein_pairs_rvs.append(pair[2] + "-" + pair[1])

        elif line[0] == '>':
            line = line.strip('\n')
            protein = line[1:]
        else:
            line = line.strip('\n')
            proteins[protein] = line
input2.close()
print(all_protein_pairs)
for i, protein in enumerate(all_protein_pairs):
    print(all_protein_pairs[i])
    print(protein_pair_labels[protein]) 

random.shuffle(all_protein_pairs)
print(all_protein_pairs)

for i, protein in enumerate(all_protein_pairs):
    print(all_protein_pairs[i])
    print(protein_pair_labels[protein])


protein_file = open('protein_amino_acid_mapping.txt', 'w+')
for key, value in proteins.items():
    protein_file.write(key + ' ' + value + '\n')

    
size_train = int(len(all_protein_pairs)*.8)
size_val = int(len(all_protein_pairs)*.1)
size_test = len(all_protein_pairs)-size_train - size_val
print("size of train", size_train)
print("size of val", size_val)
    
train_set = {}
val_set = {}
test_set = {}

train_file = open('train_file.txt', 'w+')
val_file = open('val_file.txt', 'w+')
test_file = open('test_file.txt', 'w+')

for idx, protein_pair in enumerate(all_protein_pairs):
    if idx < size_train:
        train_set[protein_pair] = protein_pair_labels[protein_pair]
        train_file.write(protein_pair + " " + str(protein_pair_labels[protein_pair]) + "\n")
    elif idx < size_train + size_val:
        val_set[protein_pair] = protein_pair_labels[protein_pair]
        val_file.write(protein_pair + " " + str(protein_pair_labels[protein_pair]) + "\n")
    else:
        train_set[protein_pair] = protein_pair_labels[protein_pair]
        train_file.write(protein_pair + " " + str(protein_pair_labels[protein_pair]) + "\n")

train_file.close()
val_file.close()
test_file.close()
protein_file.close()
