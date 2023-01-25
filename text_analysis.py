import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import csv
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import string
# nltk.download('punkt') done once
# nltk.download('stopwords') done once

## Reading and storing the given file(stop words, positive and negative words) information in list ##
def get_stop_words():
    pos_words_file_location = os.path.join(os.getcwd(), "StopWords")
    list_of_stop_word_files = os.listdir(pos_words_file_location)
    combined_stop_word_list = []
    for files in list_of_stop_word_files:
        file_name = os.path.join(os.getcwd(), "StopWords", files)
        file_data = open(file_name, "r").readlines()
        for data in file_data:
            if("|" in data):
                data = data.split("|")
                combined_stop_word_list.extend([data[0].strip(), data[1].strip()])
            else:
                combined_stop_word_list.append(data.strip())
    return combined_stop_word_list

def get_master_dict_pos_words():
    pos_words_file = os.path.join(os.getcwd(), "MasterDictionary", "positive-words.txt")
    file_content = open(pos_words_file,"r").readlines()
    pos_word_list = [word.strip() for word in file_content]
    return pos_word_list

def get_master_dict_neg_words():
    neg_words_file = os.path.join(os.getcwd(), "MasterDictionary", "negative-words.txt")
    file_content = open(neg_words_file,"r").readlines()
    neg_word_list = [word.strip() for word in file_content]
    return neg_word_list

## Read input excel file ##
def read_input_file():
    input_file_path_name = os.path.join(os.getcwd(), "Input.xlsx")
    df = pd.read_excel(input_file_path_name)
    return df

## Extracting title and article content using beautifulsoup and requests ##
def extract_article_text(urls):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0',}
    raw_html = requests.get(urls, headers=headers)
    soup = BeautifulSoup(raw_html.content, 'html.parser')
    try:
        article_title = soup.select('h1.entry-title')[0].text.strip()
        article_text = soup.find('div', class_='td-post-content').text.strip()
    except:
        article_title = ""
        article_text = ""
    article_content = article_title + " " + article_text
    return f"{article_content}"

## Saving the scraped information in text file ##
def text_file_save_path():
    file_path = os.path.join(os.getcwd(), "Extracted text files")
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    return file_path

def file_naming(url_id):
    return str(int(url_id)) + '.txt'

def write_text_file(file_name_write, output_text):
    file = open(file_name_write, "w", encoding='utf-8')
    file.write(output_text)
    file.close()
    return file.name

def save_scarped_data_in_txt_file(df_input_data):
    for url_id, urls in df_input_data.itertuples(index=False):
        output_text = extract_article_text(urls)
        file_save_path = text_file_save_path()
        file_name = file_naming(url_id)
        file_name_write = os.path.join(file_save_path, file_name)
        file_name = write_text_file(file_name_write, output_text)
    return file_save_path

## Cleaning using stop words list ##
def remove_punc(string):
    punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
    for element in string:  
        if element in punc:  
            string = string.replace(element, "") 
    return string

def get_cleaned_token_list(tokenzied_list):
    stop_word_list = get_stop_words()
    final_stop_word_list = [token.lower() for token in stop_word_list]
    tokenzied_list = [remove_punc(data) if "-" not in data else data for data in tokenzied_list]
    tokenzied_list = [data for data in tokenzied_list if(data)]
    cleaned_tokenzied_list_final = [token for token in tokenzied_list if token not in final_stop_word_list]
    return cleaned_tokenzied_list_final

## Extracting derived variables ##
def calculate_positive_score(tokenzied_list):
    pos_word_list = get_master_dict_pos_words()
    pos_word_list = [word.lower() for word in pos_word_list]  
    postive_score = 0
    for token in tokenzied_list:
        if token.lower() in pos_word_list:
            postive_score = postive_score + 1
    return postive_score

def calculate_negative_score(tokenzied_list):
    neg_word_list = get_master_dict_neg_words()
    neg_word_list = [word.lower() for word in neg_word_list]
    negative_score = 0
    for token in tokenzied_list:
        if token.lower() in neg_word_list:
            negative_score = negative_score + (-1)
    final_negative_score = negative_score * (-1)
    return final_negative_score

def calculate_polarity_score(positive_score, negative_score):
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    if(polarity_score > -1 and polarity_score < 1):
        print("polarity score coming in the range")
    else:
        print("Check. Polarity score not coming in range")
    return polarity_score

def calculate_subjectivity_score(positive_score, negative_score, total_words_after_cleaning):
    subjectivity_score = (positive_score + negative_score) / (total_words_after_cleaning + 0.000001)
    if(subjectivity_score > 0 and subjectivity_score < 1):
        print("subjectivity score coming in the range")
    else:
        print("Check. subjectivity score not coming in range")
    return subjectivity_score

def compute_derived_variables(cleaned_tokenzied_list):
    positive_score = calculate_positive_score(cleaned_tokenzied_list)
    negative_score = calculate_negative_score(cleaned_tokenzied_list)
    polarity_score = calculate_polarity_score(positive_score, negative_score)
    subjectivity_score = calculate_subjectivity_score(positive_score, negative_score, len(cleaned_tokenzied_list))
    return positive_score, negative_score, polarity_score, subjectivity_score

## Analysis of Readability ##   
def compute_avg_sentence_length(file_content, cleaned_tokenzied_list):
    sentence_list = nltk.sent_tokenize(file_content) ## splitting the sentence
    word_count = len(cleaned_tokenzied_list)
    sentence_count = len(sentence_list)
    avg_sentence_length = word_count if word_count != 0 else 0 / sentence_count if sentence_count != 0 else 0
    return avg_sentence_length

def compute_complex_word(cleaned_tokenzied_list):
    vowels = ["a","e","i","o","u"]
    list_complex_words = []
    for token in cleaned_tokenzied_list:
        count_var = 0
        for char in token.lower():
            if char in vowels:
                count_var = count_var + 1
                if(count_var > 2):
                    list_complex_words.append(token)
    return list_complex_words

def compute_percentage_of_complex_words(cleaned_tokenzied_list):
    list_complex_words = compute_complex_word(cleaned_tokenzied_list)
    total_no_complex_words = len(list_complex_words)
    total_no_of_words = len(cleaned_tokenzied_list)
    per_of_complex_words = total_no_complex_words if total_no_complex_words != 0 else 0 / total_no_of_words if total_no_of_words != 0 else 0
    return per_of_complex_words

def compute_fog_index(avg_sentence_length, per_of_complex_words):
    return 0.4 * (avg_sentence_length + per_of_complex_words)

def readability_analysis(file_content, cleaned_tokenzied_list):
    avg_sentence_length = compute_avg_sentence_length(file_content, cleaned_tokenzied_list)
    per_of_complex_words = compute_percentage_of_complex_words(cleaned_tokenzied_list)
    fog_index = compute_fog_index(avg_sentence_length, per_of_complex_words)
    return avg_sentence_length, per_of_complex_words, fog_index

## Average Number of Words Per Sentence ##
def compute_avg_no_of_words_per_sentence(file_content, cleaned_tokenzied_list):
    total_no_of_words = len(cleaned_tokenzied_list)
    sentence_list = nltk.sent_tokenize(file_content)
    sentence_count = len(sentence_list)
    avg_no_of_words_per_sentence = total_no_of_words if total_no_of_words != 0 else 0 / sentence_count if sentence_count != 0 else 0
    return avg_no_of_words_per_sentence

## Complex word count ##
def compute_complex_word_count(cleaned_tokenzied_list):
    complex_word_list = compute_complex_word(cleaned_tokenzied_list)
    return len(complex_word_list)

## Word Count ##
def compute_word_count(cleaned_tokenzied_list):
    word_count = 0
    stopWords = set(stopwords.words('english'))
    stopWords = [word.lower() for word in stopWords]
    punctuation_list = [remove_punc(word) for word in cleaned_tokenzied_list]
    # import pdb;pdb.set_trace()
    for word in punctuation_list:
        if word.lower() not in stopWords:
            word_count = word_count + 1 
    return word_count

## Syllable Count Per Word ##
def compute_syllable_count_per_word(cleaned_tokenzied_list):
    vowels = ["a","e","i","o","u"]
    count_var = 0
    for token in cleaned_tokenzied_list:
        token = token.lower()
        for char_idx in range(0,len(token)-1,1):
            if token[char_idx].isalpha() == True:
                if token[char_idx] in vowels:
                    if(token[char_idx] == "e" and (token[char_idx+1] != "d" or token[char_idx+1] != "s")):
                        count_var = count_var + 1
    if(len(cleaned_tokenzied_list) == 0 and count_var == 0):
        avg_syllable_count_per_word = 0
    else:
        avg_syllable_count_per_word = count_var / len(cleaned_tokenzied_list)
    return avg_syllable_count_per_word

## Personal Pronouns ##
def compute_personal_pronouns(file_content):
    tokenizer = RegexpTokenizer(r'\w+')
    tokenzied_list = tokenizer.tokenize(file_content)
    personal_pronouns_list = ["I", "we", "my", "ours", "us"]
    count_of_personal_pronouns = 0
    for token in tokenzied_list:
        if token in personal_pronouns_list:
            count_of_personal_pronouns = count_of_personal_pronouns + 1
    return count_of_personal_pronouns

## Average word length ##
def compute_avg_word_length(cleaned_tokenzied_list):
    total_count_char_in_eac_word = 0
    for word in cleaned_tokenzied_list:
        no_of_characters_in_word = len(word.strip())
        total_count_char_in_eac_word += no_of_characters_in_word
    if(total_count_char_in_eac_word == 0 and len(cleaned_tokenzied_list) == 0):
        avg_word_length = 0
    else:
        avg_word_length = total_count_char_in_eac_word / len(cleaned_tokenzied_list)
    return avg_word_length

## Text analysis ##
def text_analysis(file_content, cleaned_tokenzied_list):
    positive_score, negative_score, polarity_score, subjectivity_score = compute_derived_variables(cleaned_tokenzied_list)
    avg_sentence_length, per_of_complex_words, fog_index = readability_analysis(file_content, cleaned_tokenzied_list)
    avg_no_of_words_per_sentence = compute_avg_no_of_words_per_sentence(file_content, cleaned_tokenzied_list)
    word_count = compute_word_count(cleaned_tokenzied_list)
    complex_word_count = compute_complex_word_count(cleaned_tokenzied_list)
    avg_syllable_count_per_word = compute_syllable_count_per_word(cleaned_tokenzied_list)
    count_of_personal_pronouns = compute_personal_pronouns(file_content)
    avg_word_length = compute_avg_word_length(cleaned_tokenzied_list)
    return (positive_score, negative_score, polarity_score, subjectivity_score,avg_sentence_length, per_of_complex_words, fog_index, 
                            avg_no_of_words_per_sentence, word_count, complex_word_count, avg_syllable_count_per_word, count_of_personal_pronouns, avg_word_length)

## Output CSV file ##
def final_output_csv(url_id, url, positive_score, negative_score, polarity_score, subjectivity_score,avg_sentence_length, per_of_complex_words, fog_index, 
                            avg_no_of_words_per_sentence, word_count, complex_word_count, avg_syllable_count_per_word, count_of_personal_pronouns, avg_word_length):
    fields = ["URL_ID", "URL", "POSITIVE SCORE", "NEGATIVE SCORE", "POLARITY SCORE", "SUBJECTIVITY SCORE", "AVG SENTENCE LENGTH", "PERCENTAGE OF COMPLEX WORDS",
                "FOG INDEX", "AVG NUMBER OF WORDS PER SENTENCE", "COMPLEX WORD COUNT", "WORD COUNT", "SYLLABLE PER WORD", "PERSONAL PRONOUNS", "AVG WORD LENGTH"]
    data_filename = os.path.join(os.getcwd(), "Output Data Structure.csv")
    with open(data_filename, 'a', newline='') as csvfile: 
        file_is_empty = os.stat(data_filename).st_size == 0
        csvwriter = csv.writer(csvfile)   
        if file_is_empty:
            csvwriter.writerow(fields)
        csvwriter.writerow([url_id, url, positive_score, negative_score, polarity_score, subjectivity_score,avg_sentence_length, per_of_complex_words, fog_index, 
                            avg_no_of_words_per_sentence, word_count, complex_word_count, avg_syllable_count_per_word, count_of_personal_pronouns, avg_word_length])
    return data_filename

## Main function ##
def main():
    df_input_data = read_input_file()
    txt_file_save_path = save_scarped_data_in_txt_file(df_input_data)
    for files in os.listdir(txt_file_save_path):
        print(files)
        ## url and url id of the input text file ##
        url_id = files.split(".")[0]
        url = df_input_data.loc[df_input_data['URL_ID'] == int(url_id), 'URL'].iloc[0]
        ## Read input text file content ##
        file_content = open(os.path.join(txt_file_save_path, files),"r", encoding='utf-8').read()
        file_content = file_content.replace('’',"")
        ## tokenizing the words from the input text file ##
        tokenzied_list = nltk.word_tokenize(file_content)
        cleaned_tokenzied_list = get_cleaned_token_list(tokenzied_list)
        ## Text Analysis ##
        (positive_score, negative_score, polarity_score, subjectivity_score,avg_sentence_length, per_of_complex_words, fog_index, 
                            avg_no_of_words_per_sentence, word_count, complex_word_count, avg_syllable_count_per_word, count_of_personal_pronouns, avg_word_length) = text_analysis(file_content, cleaned_tokenzied_list)
        ## Writing the final output csv file ##
        final_output_file_name = final_output_csv(url_id, url, positive_score, negative_score, polarity_score, subjectivity_score,avg_sentence_length, per_of_complex_words, fog_index, 
                            avg_no_of_words_per_sentence, word_count, complex_word_count, avg_syllable_count_per_word, count_of_personal_pronouns, avg_word_length)
    return final_output_file_name


if __name__ == "__main__":
    final_output_file_name = main()
    print("Final output csv file path:\n", final_output_file_name)

    
    
