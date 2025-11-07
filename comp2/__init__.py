# ENABLE_CHECK50_ASSERT = 1
import check50
import check50.c
import check50.config

check50.config.set_truncate_len(50)

@check50.check()
def exists():
    """scoring.cc exists"""
    check50.exists("scoring.cc")

@check50.check(exists)
def compiles():
    """scoring.cc compiles"""
    check50.run("make scoring").exit(0)
    check50.exists("scoring.o")

@check50.check(compiles)
def same_user():
    """check_compatibility returns false for the same user"""
    check50.run("make 1").exit(0)

@check50.check(compiles)
def different_colleges():
    """check_compatibility returns false for users attending different colleges"""
    check50.run("make 2").exit(0)

@check50.check(compiles)
def match_type():
    """check_compatibility respects love/friendship preferences"""
    check50.run("make 3").exit(0)

@check50.check(compiles)
def mutual_crushes():
    """check_compatibility returns true for mutual crushes"""
    check50.run("make 4").exit(0)
    
@check50.check(compiles)
def class_year_gaps():
    """check_compatibility respects class year gaps"""
    check50.run("make 5").exit(0)

@check50.check(compiles)
def age_preferences():
    """check_compatibility respects age preferences"""
    check50.run("make 6").exit(0)

@check50.check(compiles)
def same_house_preferences():
    """check_compatibility respects same-house preferences"""
    check50.run("make 7").exit(0)

@check50.check(compiles)
def blocked_house_preferences():
    """check_compatibility respects blocked house preferences"""
    check50.run("make 8").exit(0)

@check50.check(compiles)
def bonus():
    """bonus function makes a meaningful contribution to score calculation"""
    # the bonus function by default only adds 0 to the score, so if the score
    # changes, then we can reasonably conclude that the bonus function has
    # probably been implemented.

    default_score = float(check50.run("make answers-9").stdout().strip("Score: "))
    your_score    = float(check50.run("make 9").stdout().strip("Score: "))
    assert abs(default_score - your_score) > 1e-3, "did you implement the bonus function?"