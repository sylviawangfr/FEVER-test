from utils.triple_extractor import printDeps, nlp, findSVOs, get_triple
from utils.file_loader import read_json_rows
import unittest
import config

class TestDB(unittest.TestCase):
    @unittest.skip(" ")
    def testSVOs(self):
        # tok = nlp("making $12 an hour? where am i going to go? i have no other financial assistance available and he certainly won't provide support.")
        # svos = findSVOs(tok)
        # printDeps(tok)
        # assert set(svos) == {('i', '!have', 'assistance'), ('he', '!provide', 'support')}
        # print(svos)

        # print("--------------------------------------------------")
        # tok = nlp("he told me i would die alone with nothing but my career someday")
        # svos = findSVOs(tok)
        # printDeps(tok)
        # print(svos)
        # assert set(svos) == {('he', 'told', 'me')}

        tok = nlp("i don't have other assistance")
        svos = findSVOs(tok)
        printDeps(tok)
        assert set(svos) == {('i', '!have', 'assistance')}

        print("-----------------------------------------------")
        tok = nlp("They ate the pizza with anchovies.")
        svos = findSVOs(tok)
        printDeps(tok)
        print(svos)
        assert set(svos) == {('they', 'ate', 'pizza'), ('they', 'ate with', 'anchovies')}

        print("--------------------------------------------------")
        tok = nlp("I have no other financial assistance available and he certainly won't provide support.")
        svos = findSVOs(tok)
        printDeps(tok)
        print(svos)
        assert set(svos) == {('i', '!have', 'assistance'), ('he', '!provide', 'support')}

        print("--------------------------------------------------")
        tok = nlp("I have no other financial assistance available, and he certainly won't provide support.")
        svos = findSVOs(tok)
        printDeps(tok)
        print(svos)
        assert set(svos) == {('i', '!have', 'assistance'), ('he', '!provide', 'support')}

        print("--------------------------------------------------")
        tok = nlp("he did not kill me")
        svos = findSVOs(tok)
        printDeps(tok)
        print(svos)
        assert set(svos) == {('he', '!kill', 'me')}

        #print("--------------------------------------------------")
        #tok = nlp("he is an evil man that hurt my child and sister")
        #svos = findSVOs(tok)
        #printDeps(tok)
        #print(svos)
        #assert set(svos) == {('he', 'hurt', 'child'), ('he', 'hurt', 'sister'), ('man', 'hurt', 'child'), ('man', 'hurt', 'sister')}

        print("--------------------------------------------------")
        tok = nlp("I wanted to kill him with a hammer.")
        svos = findSVOs(tok)
        printDeps(tok)
        print(svos)
        assert set(svos) == {('i', 'kill', 'him'), ('i', 'kill with', 'hammer')}

        print("--------------------------------------------------")
        tok = nlp("because he hit me and also made me so angry i wanted to kill him with a hammer.")
        svos = findSVOs(tok)
        printDeps(tok)
        print(svos)
        assert set(svos) == {('he', 'hit', 'me'), ('i', 'kill', 'him'), ('i', 'kill with', 'hammer')}

        print("--------------------------------------------------")
        tok = nlp("he and his brother shot me")
        svos = findSVOs(tok)
        printDeps(tok)
        print(svos)
        assert set(svos) == {('he', 'shot', 'me'), ('brother', 'shot', 'me')}

        print("--------------------------------------------------")
        tok = nlp("he and his brother shot me and my sister")
        svos = findSVOs(tok)
        printDeps(tok)
        print(svos)
        assert set(svos) == {('he', 'shot', 'me'), ('he', 'shot', 'sister'), ('brother', 'shot', 'me'), ('brother', 'shot', 'sister')}

        print("--------------------------------------------------")
        tok = nlp("the annoying person that was my boyfriend hit me")
        svos = findSVOs(tok)
        printDeps(tok)
        print(svos)
        assert set(svos) == {('person', 'was', 'boyfriend'), ('person', 'hit', 'me')}

        print("--------------------------------------------------")
        tok = nlp("the boy raced the girl who had a hat that had spots.")
        svos = findSVOs(tok)
        printDeps(tok)
        print(svos)
        assert set(svos) == {('boy', 'raced', 'girl'), ('who', 'had', 'hat'), ('hat', 'had', 'spots')}

        print("--------------------------------------------------")
        tok = nlp("he spit on me")
        svos = findSVOs(tok)
        printDeps(tok)
        print(svos)
        assert set(svos) == {('he', 'spit on', 'me')}

        print("--------------------------------------------------")
        tok = nlp("he didn't spit on me")
        svos = findSVOs(tok)
        printDeps(tok)
        print(svos)
        assert set(svos) == {('he', '!spit on', 'me')}

        print("--------------------------------------------------")
        tok = nlp("the boy raced the girl who had a hat that didn't have spots.")
        svos = findSVOs(tok)
        printDeps(tok)
        print(svos)
        assert set(svos) == {('boy', 'raced', 'girl'), ('who', 'had', 'hat'), ('hat', '!have', 'spots')}

        print("--------------------------------------------------")
        tok = nlp("he is a nice man that didn't hurt my child and sister")
        svos = findSVOs(tok)
        printDeps(tok)
        print(svos)
        assert set(svos) == {('he', 'is', 'man'), ('man', '!hurt', 'child'), ('man', '!hurt', 'sister')}

        print("--------------------------------------------------")
        tok = nlp("he didn't spit on me and my child")
        svos = findSVOs(tok)
        printDeps(tok)
        print(svos)
        assert set(svos) == {('he', '!spit on', 'me'), ('he', '!spit on', 'child')}

        print("--------------------------------------------------")
        tok = nlp("he beat and hurt me")
        svos = findSVOs(tok)
        printDeps(tok)
        print(svos)