import firebase_admin.db
import nltk
from neo4j import GraphDatabase
from nltk import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
##############################################################################################


URI = "Address"
AUTH = ("neo4j", "Password")


##############################################################################################
def initialize_firebase():
    cred = credentials.Certificate('firebase_certificate.json')
    firebase_admin.initialize_app(cred)

    db = firestore.client()
    return db


##############################################################################################
db = initialize_firebase()


##############################################################################################
def get_entity_label_firebase(entity_name):
    entity_label_ref = db.collection('labels')
    doc_ref = entity_label_ref.document(entity_name.upper())
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict()
    else:
        return None


##############################################################################################
def store_entity_label(name, label):
    entity_label_ref = db.collection('labels')
    entity_label_ref.document(name.upper()).set({'label': label})


##############################################################################################
def create_node(name, label):
    driver = GraphDatabase.driver(URI, auth=AUTH)
    summary = driver.execute_query(
        f"""
                    MERGE ({name}:{label}:SEM {{name: $name}})
                """,
        name=name,
        database_="neo4j"
    ).summary
    print("Created {nodes_created} nodes in {time} ms.".format(
        nodes_created=summary.counters.nodes_created,
        time=summary.result_available_after
    ))
    driver.close()


##############################################################################################
def clean_relation_name(relation):
    relation = relation.replace(',', ' ').replace('.', ' ')
    cleaned_relation = '_'.join(relation.split())
    return cleaned_relation


##############################################################################################
def create_relation(node1, node2, relation):
    relation = clean_relation_name(relation)
    driver = GraphDatabase.driver(URI, auth=AUTH)
    summary = driver.execute_query(
        f"""
                    MATCH ({node1} {{name:"{node1}"}}), ({node2} {{name:"{node2}"}})
                    MERGE ({node1})-[:{relation}]->({node2})
                    """,
        database_="neo4j",
    ).summary
    if summary.counters.relationships_created > 0:
        print(f"Created a relationship of type '{relation}' between '{node1}' and '{node2}'.")
    else:
        print(f"No relationship created between '{node1}' and '{node2}'. This might be because the nodes don't exist or already have the specified relationship.")
    driver.close()


##############################################################################################
def create_semantic_graph(text):
    named_entities, pos_tagged = create_named_entities(text)
    for word in pos_tagged:
        node_label = None
        if word[1] == "NNP":                            ##entity detected
            for entity in named_entities:
                if word[0] in entity[1][0]:             ##checking whether nertag detected the entity or not
                    node_name = word[0]
                    node_label = entity[0]
                    create_node(node_name, node_label)
                    store_entity_label(node_name, node_label)
                    break
        if word[1] == "NNP" and node_label is None:     ##checking in database
            node_name = word[0]
            db_label = get_entity_label_firebase(node_name)
            if db_label is not None:
                node_label = db_label['label']
                create_node(node_name, node_label)
        if node_label is None and word[1] == "NNP":     ##asking user
            node_label = input(f"Who/What is {word[0]}: ")
            node_name = word[0]
            create_node(node_name, node_label)
            store_entity_label(node_name, node_label)

    entity_count = 0
    word_stack = []                             ##using stack to store words, it will help in making relations, when second entity is detected, the words between the second entity to first entity becomes relation words.
    relation_list = []
    for word in pos_tagged:
        word_stack.append(word)
        if word[1] == "NNP":
            entity_count = entity_count + 1
        if entity_count == 2:
            entity_count = 1
            second_entity = word_stack.pop()
            first_entity = word
            while True:
                i = word_stack.pop()
                if i[1] == "NNP":
                    first_entity = i
                    break
                relation_list.append(i)
            relation = ""
            relation_list.reverse()
            for relation_word in relation_list:
                relation = relation + "_" + relation_word[0]
            create_relation(first_entity[0], second_entity[0], relation)
            word_stack.append(second_entity)
            relation_list.clear()


##############################################################################################
def create_named_entities(text):
    tokens = word_tokenize(text)
    words = []
    Lemmatizer = WordNetLemmatizer()
    for token in tokens:
        words.append(Lemmatizer.lemmatize(token))
    print(words)
    p_tagged = pos_tag(words)
    print(p_tagged)
    named_entities = nltk.ne_chunk(p_tagged, binary=False)
    entities = []
    for entity in named_entities:
        if hasattr(entity, "label"):
            label = entity.label()
            entities.append((label, entity.leaves()))
    print(entities)
    return entities, p_tagged
##############################################################################################


text = input("Enter a phrase: ")
create_semantic_graph(text)
