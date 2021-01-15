#include "helper.h"

void testBasic()
{
  auto foo = 2.0;
  auto bar = 1.0;

  ALEPH_ASSERT_THROW( foo != bar );
  ALEPH_ASSERT_EQUAL( foo, 2.0 );
  ALEPH_ASSERT_EQUAL( bar, 1.0 );
}

void testAdvanced()
{
  // a more advanced test
}

int main(int, char**)
{
  testBasic();
  testAdvanced();
}