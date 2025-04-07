import csv
from collections import defaultdict
import chardet
from PIL import Image
import os

# 使用Pillow库加载一张图片，并将其转换为RGB模式
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
# 找到txt文件中文本描述部分
def mimix_reports_loader(path_reports):
    with open(path_reports, 'r') as file:
        content = file.read()
        start_index_findings = content.find('FINDINGS:') + len('FINDINGS:')
        end_index_findings = content.find('IMPRESSION:')
        findings_text = content[start_index_findings:end_index_findings].replace('\n', '').strip()
 
    return findings_text

class DatasetProcessor:
    def __init__(self, metadata='mimic-cxr-2.1.0-split.csv',tag = 'train'):
        self.metadata = metadata
        self.samples = []
        self.tag = 'train'
 
    def process_dataset(self):
        samples = defaultdict(list)
        with open(metadata, 'rb') as f:
        # 使用chardet检测文件编码格式
            file_content = f.read()
            encoding = chardet.detect(file_content)['encoding']
        with open(self.metadata, mode='r', encoding=encoding) as f:
            reader = csv.DictReader((line.replace('\0', '') for line in f))
            print(reader.fieldnames)  # 打印列名
            for row in reader:
                if row['split'] == self.tag:  # 仅处理split为train的样本
                    key = (row['dicom_id'], row['study_id'], row['subject_id'], row['split'])
                    samples[key].append(None)  # 添加一个空值作为占位符
                
            self.samples = list(samples.keys())  # 仅保存键值对的键

if __name__ == "__main__":
    processor = DatasetProcessor('mimic-cxr-2.1.0-split.csv','train')
    # 处理数据集
    processor.process_dataset()
    
    # 输出
    for i in range(len(processor.samples)):
        dicom_id, study_id, subject_id, split = processor.samples[i]
        print(f"dicom_id: {dicom_id}, study_id: {study_id}, subject_id: {subject_id}, split: {split}")
        
        path = os.path.join(root, 'mimic-cxr-images/files', f'p{subject_id[:2]}', f'p{subject_id}', f's{study_id}', f"{dicom_id}.jpg")
        img = pil_loader(path)
        img.show()  # 显示图片
        path_reports = os.path.join(root, 'mimic-cxr-reports/files', f'p{subject_id[:2]}', f'p{subject_id}', f"s{study_id}.txt")
        caption = mimix_reports_loader(path_reports)
        print(caption)