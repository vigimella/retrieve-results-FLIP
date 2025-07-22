import os, re, csv, argparse

def retrieve_results(file_path):

  keywords = ["model:", "dataset:", "distribution:", "img_size:", "total_clients", "clients_per_round", "total_rounds", "client_epochs_per_round", "learning_rate"]

  specific_lines = []

  specific_lines.append(f'Path: {file_path}')

  test_metrics_pattern = r"Test Metrics:\n\n(.*?)(?=\n\n Confusion Matrix)"
  execution_time_pattern = r"Execution Time: (\d{2}:\d{2}:\d{2})"

  hyper_parameters = list()

  with open(file_path, "r") as file_:

      for line in file_:
          if any(keyword in line for keyword in keywords):
              specific_lines.append(line.strip().replace('_', ' ').title())

  with open(file_path, "r") as file_p:
      content = file_p.read()

  test_metrics_pattern = r"Test Metrics:\n\n(.*?)(?=\n\n Confusion Matrix)"
  execution_time_pattern = r"Execution Time: (\d{2}:\d{2}:\d{2})"

  test_metrics = re.search(test_metrics_pattern, content, re.S)
  execution_time = re.search(execution_time_pattern, content, re.S)

  value_precision_to_remove = re.sub(r"(: )\d+\.\d+", r"\1", test_metrics.group(1).strip().split('\n')[3])
  value_recall_to_remove = re.sub(r"(: )\d+\.\d+", r"\1", test_metrics.group(1).strip().split('\n')[4])
  value_auc_to_remove = re.sub(r"(: )\d+\.\d+", r"\1", test_metrics.group(1).strip().split('\n')[5])

  loss = test_metrics.group(1).strip().split('\n')[1].replace('loss: ', '')
  accuracy = test_metrics.group(1).strip().split('\n')[2].replace('categorical_accuracy: ', '')
  precision = test_metrics.group(1).strip().split('\n')[3].replace(value_precision_to_remove, '')
  recall = test_metrics.group(1).strip().split('\n')[4].replace(value_recall_to_remove, '')
  auc = test_metrics.group(1).strip().split('\n')[5].replace(value_auc_to_remove, '')
  fmeasure = test_metrics.group(1).strip().split('\n')[6].replace('fmeasure: ', '')

  specific_lines.append(f'Loss: {loss}')
  specific_lines.append(f'Accuracy: {accuracy}')
  specific_lines.append(f'Precision: {precision}')
  specific_lines.append(f'recall')
  specific_lines.append(f'Recall: {recall}')
  specific_lines.append(f'F-Measure: {fmeasure}')
  specific_lines.append(f'AUC: {auc}')

  return specific_lines

def create_csv(metrics, csv_name):

    all_metrics_data = []

    for metric_set in metrics:
        metrics_data = []
        for line in metric_set:
            parts = line.split(": ")
            if len(parts) == 2:
                metric_name, metric_value = parts
                metrics_data.append(metric_value.strip())

        all_metrics_data.append(metrics_data)

    with open(csv_name, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        header = [line.split(": ")[0].strip() for line in metrics[0] if ": " in line]
        writer.writerow(header)

        writer.writerows(all_metrics_data)

    print(f"CSV created: {csv_name}")

def retrieve_folders_path(main_folder):

  sub_folder = list()

  for folder in os.listdir(main_folder):
    folder = os.path.join(main_folder, folder)
    
    if os.path.isdir(folder):
        sub_folder.append(folder)

  return sub_folder

def retrieve_file_path(sub_folder):

  complete_list = list()

  for sub_folder_ in sub_folder:

    for file in os.listdir(sub_folder_):

      if file.endswith('.txt'):

        file_path = os.path.join(sub_folder_, file)
        complete_list.append(retrieve_results(file_path))

  return complete_list

def parse_args():
    parser = argparse.ArgumentParser(prog="python3.9 main.py",
                                     description='retrieve results obtained using Federated Learning for Image Processing')
    group = parser.add_argument_group('Arguments')
    group.add_argument('-i', '--inputfolder', required=True, type=str, default=None,
                       help='the folder path where results are stored')
    group.add_argument('-n', '--namecsv', required=True, type=str, default=None,
                       help='the csv name')
    arguments = parser.parse_args()
    return arguments

if __name__ == "__main__":
  
  args = parse_args()
  folders_path = retrieve_folders_path(args.inputfolder)
  complete_list = retrieve_file_path(folders_path)

  if args.namecsv.endswith('.csv'):
      create_csv(complete_list, args.namecsv)
  else:
      create_csv(complete_list, args.namecsv + '.csv')