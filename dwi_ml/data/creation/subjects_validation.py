
# Fixme
# NE SERA PLUS UTILE SI ON GERE COMME IL FAUT LES dataset_creation (hdf5).
# A GERER AVANT DE MODIFIER CE FICHIER


def list_intersection(list1, list2):  # or list(set(a) & set(b))
    """Values both in list1 and list2"""
    list3 = [value for value in list1 if value in list2]
    return list3


def list_difference(list1, list2):
    """Values in list1 that are not in list2 = list1 - list2"""
    list3 = [value for value in list1 if value not in list2]
    return list3


def list_equals(list1, list2):
    """Compares two lists"""
    list1.sort()                                                                                # might bug if some subjs are numeric, other alpha numeric. We suppose all alphanumeric?
    list2.sort()
    return list1 == list2


def validate_subject_list(all_subjects, chosen):
    """
    Parameters
    ----------
    List1 = all subjects.
    List2 = some chosen subjects.

    Returns
    --------
    error_subjects: chosen subjects who don't exist
    good_subjects: chosen subjects who exist
    forgotten_subjects: subjects who exists but are not chosen
    """
    error_subjects = list_difference(chosen, all_subjects)
    good_subjects = list_intersection(all_subjects, chosen)
    forgotten_subjects = list_difference(all_subjects, chosen)

    return error_subjects, good_subjects, forgotten_subjects
