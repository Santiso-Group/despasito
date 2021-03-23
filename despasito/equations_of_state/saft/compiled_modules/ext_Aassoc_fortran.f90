subroutine calc_Xika_inner_4(Xika0,indices,xi,rho,nui,nk,Fklab,Kklab,Iij,obj_func,Xika,ncomp,nbeads,nsitemax,nind)
    implicit none
    integer,intent(in) :: ncomp, nbeads, nsitemax, nind
    real(8), intent(in), dimension(0:nind-1) :: Xika0
    real(8), intent(in), dimension(0:nind-1, 0:2) :: indices
    real(8), intent(in), dimension(0:ncomp-1) :: xi
    real(8), intent(in), dimension(0:ncomp-1,0:nbeads-1) :: nui
    real(8), intent(in), dimension(0:nbeads-1,0:nsitemax-1) :: nk
    real(8), intent(in), dimension(0:nbeads-1,0:nbeads-1,0:nsitemax-1,0:nsitemax-1) :: Fklab
    real(8), intent(in), dimension(0:nbeads-1,0:nbeads-1,0:nsitemax-1,0:nsitemax-1) :: Kklab
    real(8), intent(in), dimension(0:ncomp-1,0:ncomp-1) :: Iij
    real(8), intent(in) :: rho
    real(8), intent(out), dimension(0:nind-1) :: obj_func
    real(8), intent(out), dimension(0:nind-1) :: Xika
    integer :: ii,jj,i,k,a,j,l,b

    !initialize Xika to 1.0
    Xika=1.0
    
    do ii=0, nind-1
        i = indices(ii,0)
        k = indices(ii,1)
        a = indices(ii,2)
        do jj=0, nind-1
            j = indices(jj,0)
            l = indices(jj,1)
            b = indices(jj,2)
            Xika(ii)=Xika(ii) + (rho * xi(j) * nui(j,l) * nk(l,b) * Xika0(jj) * Fklab(k,l,a,b) * Kklab(k,l,a,b) * Iij(i,j))
        enddo
    enddo
    Xika = (1.0/Xika)
    
    obj_func = Xika - Xika0
    
end subroutine


subroutine calc_Xika_4(indices,rho,Xika_init,xi,nui,nk,Fklab,Kklab,Iij,maxiter,tol,Xika_final,nrho,ncomp,nbeads,nsitemax,nind)
    implicit none
    integer, intent(in) :: nrho, ncomp, nbeads, nsitemax, nind
    real(8), intent(in), dimension(0:nind-1, 0:2) :: indices
    real(8), intent(in), dimension(0:nrho-1) :: rho
    real(8), intent(in), dimension(0:nind-1) :: Xika_init
    real(8), intent(in), dimension(0:ncomp-1) :: xi
    real(8), intent(in), dimension(0:ncomp-1,0:nbeads-1) :: nui
    real(8), intent(in), dimension(0:nbeads-1,0:nsitemax-1) :: nk
    real(8), intent(in), dimension(0:nbeads-1,0:nbeads-1,0:nsitemax-1,0:nsitemax-1) :: Fklab
    real(8), intent(in), dimension(0:nbeads-1,0:nbeads-1,0:nsitemax-1,0:nsitemax-1) :: Kklab
    real(8), intent(in), dimension(0:nrho-1,0:ncomp-1,0:ncomp-1) :: Iij
    integer, intent(in) :: maxiter
    real(8), intent(in) :: tol
    real(8), intent(out), dimension(0:nrho-1,0:nind-1) :: Xika_final
    real(8), dimension(0:nind-1) :: Xika,obj_func,Xika0
    integer :: i,iter
    
    
    Xika=Xika_init
    
    do i=0, nrho-1
        !write(*,*) Xika0
        do iter=0, maxiter-1
            Xika0=Xika
            call calc_Xika_inner_4(Xika0,indices,xi,rho(i),nui,nk,Fklab,Kklab,Iij(i,:,:),obj_func,Xika,ncomp,nbeads,nsitemax,nind)            
            if(maxval(dabs(obj_func)).lt.tol) exit
        enddo
        Xika_final(i,:)=Xika
        !write(*,*) i,iter,Xika,obj_func
    end do
    !write(*,*) i,rho(i),Xika,Xika_final(i,:,:,:)
end subroutine

subroutine calc_Xika_inner_6(Xika0,indices,xi,rho,nui,nk,Fklab,Kklab,Iij,obj_func,Xika,ncomp,nbeads,nsitemax,nind)
    implicit none
    integer,intent(in) :: ncomp, nbeads, nsitemax, nind
    real(8), intent(in), dimension(0:nind-1) :: Xika0
    real(8), intent(in), dimension(0:nind-1, 0:2) :: indices
    real(8), intent(in), dimension(0:ncomp-1) :: xi
    real(8), intent(in), dimension(0:ncomp-1,0:nbeads-1) :: nui
    real(8), intent(in), dimension(0:nbeads-1,0:nsitemax-1) :: nk
    real(8), intent(in), dimension(0:nbeads-1,0:nbeads-1,0:nsitemax-1,0:nsitemax-1) :: Fklab
    real(8), intent(in), dimension(0:ncomp-1,0:ncomp-1,0:nbeads-1,0:nbeads-1,0:nsitemax-1,0:nsitemax-1) :: Kklab
    real(8), intent(in), dimension(0:ncomp-1,0:ncomp-1) :: Iij
    real(8), intent(in) :: rho
    real(8), intent(out), dimension(0:nind-1) :: obj_func
    real(8), intent(out), dimension(0:nind-1) :: Xika
    integer :: ii,jj,i,k,a,j,l,b

    !initialize Xika to 1.0
    Xika=1.0

    do ii=0, nind-1
        i = indices(ii,0)
        k = indices(ii,1)
        a = indices(ii,2)
        do jj=0, nind-1
            j = indices(jj,0)
            l = indices(jj,1)
            b = indices(jj,2)
            Xika(ii)=Xika(ii) + (rho * xi(j) * nui(j,l) * nk(l,b) * Xika0(jj) * Fklab(k,l,a,b) * Kklab(i,j,k,l,a,b) * Iij(i,j))
        enddo
    enddo
    Xika = (1.0/Xika)

    obj_func = Xika - Xika0

end subroutine


subroutine calc_Xika_6(indices,rho,Xika_init,xi,nui,nk,Fklab,Kklab,Iij,maxiter,tol,Xika_final,nrho,ncomp,nbeads,nsitemax,nind)
    implicit none
    integer, intent(in) :: nrho, ncomp, nbeads, nsitemax, nind
    real(8), intent(in), dimension(0:nind-1, 0:2) :: indices
    real(8), intent(in), dimension(0:nrho-1) :: rho
    real(8), intent(in), dimension(0:nind-1) :: Xika_init
    real(8), intent(in), dimension(0:ncomp-1) :: xi
    real(8), intent(in), dimension(0:ncomp-1,0:nbeads-1) :: nui
    real(8), intent(in), dimension(0:nbeads-1,0:nsitemax-1) :: nk
    real(8), intent(in), dimension(0:nbeads-1,0:nbeads-1,0:nsitemax-1,0:nsitemax-1) :: Fklab
    real(8), intent(in), dimension(0:ncomp-1,0:ncomp-1,0:nbeads-1,0:nbeads-1,0:nsitemax-1,0:nsitemax-1) :: Kklab
    real(8), intent(in), dimension(0:nrho-1,0:ncomp-1,0:ncomp-1) :: Iij
    integer, intent(in) :: maxiter
    real(8), intent(in) :: tol
    real(8), intent(out), dimension(0:nrho-1,0:nind-1) :: Xika_final
    real(8), dimension(0:nind-1) :: Xika,obj_func,Xika0
    integer :: i,iter


    Xika=Xika_init

    do i=0, nrho-1
        !write(*,*) Xika0
        do iter=0, maxiter-1
            Xika0=Xika
            call calc_Xika_inner_6(Xika0,indices,xi,rho(i),nui,nk,Fklab,Kklab,Iij(i,:,:),obj_func,Xika,ncomp,nbeads,nsitemax,nind)
            if(maxval(dabs(obj_func)).lt.tol) exit
        enddo
        Xika_final(i,:)=Xika
        !write(*,*) i,iter,Xika,obj_func
    end do
    !write(*,*) i,rho(i),Xika,Xika_final(i,:,:,:)
end subroutine

