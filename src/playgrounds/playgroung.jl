using CUDA

codee = """
define dso_local i16 @"?addToTp@@YAG_N0G@Z"(i1 zeroext %0, i1 zeroext %1, i16 %2) #0 {
  %4 = alloca i16, align 2
  %5 = alloca i8, align 1
  %6 = alloca i8, align 1
  store i16 %2, i16* %4, align 2
  %7 = zext i1 %1 to i8
  store i8 %7, i8* %5, align 1
  %8 = zext i1 %0 to i8
  store i8 %8, i8* %6, align 1
  %9 = load i8, i8* %6, align 1
  %10 = trunc i8 %9 to i1
  %11 = zext i1 %10 to i32
  %12 = load i8, i8* %5, align 1
  %13 = trunc i8 %12 to i1
  %14 = zext i1 %13 to i32
  %15 = and i32 %11, %14
  %16 = load i16, i16* %4, align 2
  %17 = zext i16 %16 to i32
  %18 = add nsw i32 %17, %15
  %19 = trunc i32 %18 to i16
  store i16 %19, i16* %4, align 2
  ret i16 %19
}
"""

fname = Symbol("addToTp")
@eval export $fname
@eval begin
	@inline addToTp(boolGold, boolSegm, tp) =
		Base.llvmcall($("""
		$codee
			attributes #0 = { alwaysinline }""", "addToTp"),
			UInt32, Tuple{Bool, Bool, UInt32}, boolGold, boolSegm, tp)
end
addToTp(true, true, UInt32(1))

using Test
function addToFp(boolGold::Bool, boolSegm::Bool, tp::UInt16)
	Base.llvmcall("""
	%4 = xor i8 %0, %1
	%5 = and i8 %4, %1
	%6 = zext i8 %5 to i16
	%7 = add i16 %2,%6
	ret i16 %7""", UInt16, Tuple{Bool, Bool, UInt16}, boolGold, boolSegm, tp)
end

@test addToFp(false, true, UInt16(2)) == UInt16(3)
@test addToFp(true, true, UInt16(2)) == UInt16(2)
@test addToFp(false, false, UInt16(2)) == UInt16(2)
@test addToFp(true, false, UInt16(2)) == UInt16(2)


function addToFn(boolGold::Bool, boolSegm::Bool, tp::UInt16)
	Base.llvmcall("""
	%4 = xor i8 %0, %1
	%5 = and i8 %4, %0
	%6 = zext i8 %5 to i16
	%7 = add i16 %2,%6
	ret i16 %7""", UInt16, Tuple{Bool, Bool, UInt16}, boolGold, boolSegm, tp)
end

@test addToFn(false, true, UInt16(2)) == UInt16(2)
@test addToFn(true, true, UInt16(2)) == UInt16(2)
@test addToFn(false, false, UInt16(2)) == UInt16(2)
@test addToFn(true, false, UInt16(2)) == UInt16(3)

x = UInt16(1)
zz(true, true, x)
x