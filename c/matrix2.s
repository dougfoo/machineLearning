	.section	__TEXT,__text,regular,pure_instructions
	.macosx_version_min 10, 13
	.globl	_main                   ## -- Begin function main
	.p2align	4, 0x90
_main:                                  ## @main
	.cfi_startproc
## BB#0:
	pushq	%rbp
Lcfi0:
	.cfi_def_cfa_offset 16
Lcfi1:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
Lcfi2:
	.cfi_def_cfa_register %rbp
	subq	$208, %rsp
	leaq	l_main.second(%rip), %rax
	movl	$36, %ecx
	movl	%ecx, %edx
	leaq	-96(%rbp), %rsi
	leaq	l_main.first(%rip), %rdi
	leaq	-48(%rbp), %r8
	movq	___stack_chk_guard@GOTPCREL(%rip), %r9
	movq	(%r9), %r9
	movq	%r9, -8(%rbp)
	movl	$0, -148(%rbp)
	movw	$0, -156(%rbp)
	movq	%rdi, -168(%rbp)        ## 8-byte Spill
	movq	%r8, %rdi
	movq	-168(%rbp), %r8         ## 8-byte Reload
	movq	%rsi, -176(%rbp)        ## 8-byte Spill
	movq	%r8, %rsi
	movq	%rdx, -184(%rbp)        ## 8-byte Spill
	movq	%rax, -192(%rbp)        ## 8-byte Spill
	callq	_memcpy
	movq	-176(%rbp), %rax        ## 8-byte Reload
	movq	%rax, %rdi
	movq	-192(%rbp), %rsi        ## 8-byte Reload
	movq	-184(%rbp), %rdx        ## 8-byte Reload
	callq	_memcpy
	movw	$0, -150(%rbp)
LBB0_1:                                 ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_3 Depth 2
                                        ##       Child Loop BB0_5 Depth 3
	movswl	-150(%rbp), %eax
	cmpl	$3, %eax
	jge	LBB0_12
## BB#2:                                ##   in Loop: Header=BB0_1 Depth=1
	movw	$0, -152(%rbp)
LBB0_3:                                 ##   Parent Loop BB0_1 Depth=1
                                        ## =>  This Loop Header: Depth=2
                                        ##       Child Loop BB0_5 Depth 3
	movswl	-152(%rbp), %eax
	cmpl	$3, %eax
	jge	LBB0_10
## BB#4:                                ##   in Loop: Header=BB0_3 Depth=2
	movw	$0, -154(%rbp)
LBB0_5:                                 ##   Parent Loop BB0_1 Depth=1
                                        ##     Parent Loop BB0_3 Depth=2
                                        ## =>    This Inner Loop Header: Depth=3
	movswl	-154(%rbp), %eax
	cmpl	$3, %eax
	jge	LBB0_8
## BB#6:                                ##   in Loop: Header=BB0_5 Depth=3
	leaq	-96(%rbp), %rax
	leaq	-48(%rbp), %rcx
	movswl	-156(%rbp), %edx
	movswq	-150(%rbp), %rsi
	imulq	$12, %rsi, %rsi
	addq	%rsi, %rcx
	movswq	-154(%rbp), %rsi
	movl	(%rcx,%rsi,4), %edi
	movswq	-154(%rbp), %rcx
	imulq	$12, %rcx, %rcx
	addq	%rcx, %rax
	movswq	-152(%rbp), %rcx
	imull	(%rax,%rcx,4), %edi
	addl	%edi, %edx
	movw	%dx, %r8w
	movw	%r8w, -156(%rbp)
## BB#7:                                ##   in Loop: Header=BB0_5 Depth=3
	movw	-154(%rbp), %ax
	addw	$1, %ax
	movw	%ax, -154(%rbp)
	jmp	LBB0_5
LBB0_8:                                 ##   in Loop: Header=BB0_3 Depth=2
	leaq	-144(%rbp), %rax
	movswl	-156(%rbp), %ecx
	movswq	-150(%rbp), %rdx
	imulq	$12, %rdx, %rdx
	addq	%rdx, %rax
	movswq	-152(%rbp), %rdx
	movl	%ecx, (%rax,%rdx,4)
	movw	$0, -156(%rbp)
## BB#9:                                ##   in Loop: Header=BB0_3 Depth=2
	movw	-152(%rbp), %ax
	addw	$1, %ax
	movw	%ax, -152(%rbp)
	jmp	LBB0_3
LBB0_10:                                ##   in Loop: Header=BB0_1 Depth=1
	jmp	LBB0_11
LBB0_11:                                ##   in Loop: Header=BB0_1 Depth=1
	movw	-150(%rbp), %ax
	addw	$1, %ax
	movw	%ax, -150(%rbp)
	jmp	LBB0_1
LBB0_12:
	movl	-148(%rbp), %eax
	movq	___stack_chk_guard@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rcx
	movq	-8(%rbp), %rdx
	cmpq	%rdx, %rcx
	movl	%eax, -196(%rbp)        ## 4-byte Spill
	jne	LBB0_14
## BB#13:
	movl	-196(%rbp), %eax        ## 4-byte Reload
	addq	$208, %rsp
	popq	%rbp
	retq
LBB0_14:
	callq	___stack_chk_fail
	.cfi_endproc
                                        ## -- End function
	.section	__TEXT,__const
	.p2align	4               ## @main.first
l_main.first:
	.long	0                       ## 0x0
	.long	1                       ## 0x1
	.long	2                       ## 0x2
	.long	3                       ## 0x3
	.long	4                       ## 0x4
	.long	5                       ## 0x5
	.long	6                       ## 0x6
	.long	7                       ## 0x7
	.long	8                       ## 0x8

	.p2align	4               ## @main.second
l_main.second:
	.long	5                       ## 0x5
	.long	5                       ## 0x5
	.long	5                       ## 0x5
	.long	4294967291              ## 0xfffffffb
	.long	4294967291              ## 0xfffffffb
	.long	4294967291              ## 0xfffffffb
	.long	5                       ## 0x5
	.long	5                       ## 0x5
	.long	5                       ## 0x5


.subsections_via_symbols
