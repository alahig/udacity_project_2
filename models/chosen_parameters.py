    # We tested wheter:

    #Tokenization:
    #- It is useful to replaced named_entities (Locations, numbers, person names) -> Answer: Yes
    #- PorterStemming improvies the prediciton -> Answer: Yes

    #Meta data of the sentence:
    #- Test for the presence of ? -> Answer: No
    #- Test for the presence of ! -> Answer: Yes
    #- Look at the distribution of words (Nouns, Verbs, ..)  -> Answer: No
    #- Look at the presence of capital letters (I NEED HELP) -> Answer: Yes

    #Model:
    #- Number of estimators to be used in the tree: 200 or 100 or 50 -> Better 50

parameters = {
    'features__nlp_pipeline__count__tokenizer__replace_named_entities': [True], #False
    'features__nlp_pipeline__count__tokenizer__use_stemming': [True], # False
    'classifier__estimator__n_estimators': [50], #100 150,
    'features__word_type_counter__use_question_mark': [False], # False
    'features__word_type_counter__use_pct_word_types': [False],  # False
    'features__word_type_counter__use_pct_capital_letters': [True],  # False
    'features__word_type_counter__use_exclamation_mark': [True],  # False
        
    }