bootstrap_servers = 'localhost:9095'
conf = {'bootstrap.servers': bootstrap_servers, 'group.id': 'my'}

metrics_topic = 'metrics'
preprocessing_topic = 'preprocessing'

annotation_path = "data/list_attr_celeba.csv"
img_dir = "data/img_align_celeba/img_align_celeba"
save_preprocessing_dir = "preprocessed_data/"