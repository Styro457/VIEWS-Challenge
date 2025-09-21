from views_challenge.utils import utils


def test_countries_decode():
    assert utils.decode_country(642) == "Romania"
    assert utils.decode_country(643) == "Russian Federation"
    assert utils.decode_country(120) == "Cameroon"
