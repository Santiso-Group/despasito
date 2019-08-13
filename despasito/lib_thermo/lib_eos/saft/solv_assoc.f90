subroutine calc_Xika(Xika0,xi,rho,nui,nk,delta,obj_func,Xika,ncomp,nbeads,nsitemax)
    implicit none
    integer,intent(in) :: ncomp, nbeads, nsitemax
    real(8), intent(in), dimension(0:ncomp-1, 0:nbeads-1, 0:nsitemax-1) :: Xika0
    real(8), intent(in), dimension(0:ncomp-1) :: xi
    real(8), intent(in), dimension(0:ncomp-1,0:nbeads-1) :: nui
    real(8), intent(in), dimension(0:nbeads-1,0:nsitemax-1) :: nk
    real(8), intent(in), dimension(0:ncomp-1,0:ncomp-1,0:nbeads-1,0:nbeads-1,0:nsitemax-1,0:nsitemax-1) :: delta
    real(8), intent(in) :: rho
    real(8), intent(out), dimension(0:ncomp-1,0:nbeads-1,0:nsitemax-1) :: obj_func
    real(8), intent(out), dimension(0:ncomp-1,0:nbeads-1,0:nsitemax-1) :: Xika
    integer :: i,k,a,j,l,b
    
    !initialize Xika to 1.0
    Xika=1.0
    
    do i=0, ncomp-1
        do k=0, nbeads-1
            do a=0, nsitemax-1
                do j=0, ncomp-1
                    do l=0, nbeads-1
                        do b=0, nsitemax-1
                            Xika(i,k,a)=Xika(i,k,a) + (rho * xi(j) * nui(j,l) * nk(l,b) * Xika0(j,l,b) * delta(i,j,k,l,a,b))
                        enddo
                    enddo
                enddo
            enddo
        enddo
    enddo
    Xika = (1.0/Xika)
    
    obj_func = Xika - Xika0
    
end subroutine


subroutine min_xika(rho,Xika_init,xi,nui,nk,delta,maxiter,tol,Xika_final,nrho,ncomp,nbeads,nsitemax)
    implicit none
    integer, intent(in) :: nrho,ncomp,nbeads,nsitemax
    real(8), intent(in), dimension(0:nrho-1) :: rho
    real(8), intent(in), dimension(0:ncomp-1, 0:nbeads-1, 0:nsitemax-1) :: Xika_init
    real(8), intent(in), dimension(0:ncomp-1) :: xi
    real(8), intent(in), dimension(0:ncomp-1,0:nbeads-1) :: nui
    real(8), intent(in), dimension(0:nbeads-1,0:nsitemax-1) :: nk
    real(8), intent(in), dimension(0:nrho-1,0:ncomp-1,0:ncomp-1,0:nbeads-1,0:nbeads-1,0:nsitemax-1,0:nsitemax-1) :: delta
    integer, intent(in) :: maxiter
    real(8), intent(in) :: tol
    real(8), intent(out), dimension(0:nrho-1,0:ncomp-1, 0:nbeads-1, 0:nsitemax-1) :: Xika_final
    real(8), dimension(0:ncomp-1, 0:nbeads-1, 0:nsitemax-1) :: Xika,obj_func,Xika0
    real(8), dimension(0:(ncomp*nbeads*nsitemax)-1) :: Xika_flat
    integer :: i,iter
    
    
    Xika=Xika_init
    
    do i=0, nrho-1
        !write(*,*) Xika0
        do iter=0, maxiter-1
            Xika0=Xika
            call calc_Xika(Xika0,xi,rho(i),nui,nk,delta(i,:,:,:,:,:,:),obj_func,Xika,ncomp,nbeads,nsitemax)            
            if(maxval(dabs(obj_func)).lt.tol) exit
        enddo
        Xika_final(i,:,:,:)=Xika
        !write(*,*) i,iter,Xika,obj_func
    end do
    !write(*,*) i,rho(i),Xika,Xika_final(i,:,:,:)
end subroutine

