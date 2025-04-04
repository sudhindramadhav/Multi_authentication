# utils/pincode.py

def verify_pincode(stored_pincode, entered_pincode):
    """
    Verifies if the entered pincode matches the stored pincode.
    
    Args:
        stored_pincode (str): The pincode stored in the database.
        entered_pincode (str): The pincode entered by the user.

    Returns:
        bool: True if the pincode matches, False otherwise.
    """
    return stored_pincode == entered_pincode
